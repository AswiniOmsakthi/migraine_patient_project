import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph

from src.config import restricted_tools
from src.custom_prompts.multi_agent_prompts import (
    fallback_invalid_response,
    fallback_response,
    system_message_generate_answer,
    system_message_query_analyzer,
    system_message_validator,
)
from src.utils.app_logger import LOGGER


class MigraineQAState(TypedDict):
    question: str
    system_message_generate_answer: str
    system_message_query_analyzer: str
    system_message_validator: str
    selected_tools_queries: Dict[str, List[str]]
    tool_descriptions: Optional[List[str]]
    context: List[str]
    answer: Optional[str]
    validated: Optional[bool]
    answer_reasoning: Optional[str]
    attempts: int
    fallback: Optional[str]
    fallback_response: Optional[str]
    memory: List[Dict[str, Any]]
    openai_llm: AzureChatOpenAI
    tool_list: List[Any]


def query_analyzer(state: MigraineQAState) -> MigraineQAState:
    start_time = time.time()
    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in state['tool_list']])
    memory_context = ""
    if state["memory"]:
        memory_context = "Previous conversation:\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["memory"]])
    system_prompt = (
        f"{state.get('system_message_query_analyzer','').format(memory_context=memory_context, tool_descriptions=tool_descriptions)}. \n."
        "Add User question as well to the appropriate tool.\n"
        "Respond in JSON as a list: DO NOT wrap json output with ```json ```"
        "[{\"tool\": <tool_name>, \"query\": [User question, search_query_1, search_query_2...] }]\n"
        "If no tool is appropriate, respond with an empty list []."
    )

    user_prompt = f"User question: {state['question']}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = state['openai_llm'].invoke(messages)

    if isinstance(response.content, str):
        state = _parse_selected_tools_and_queries(response.content, state)
        LOGGER.info(response)
    else:
        state["fallback"] = "clarification"
        state["fallback_response"] = fallback_response
        LOGGER.warning("No tools found")
    LOGGER.info(f"Query Analyzer Time, {time.time() - start_time}")
    return state


def _parse_selected_tools_and_queries(json_string: str, state: MigraineQAState) -> MigraineQAState:
    try:
        result = json.loads(json_string)
        if not isinstance(result, list):
            raise ValueError("Expected list of tool-query dicts")
        state["selected_tools_queries"] = {}
        state["tool_descriptions"] = []
        for entry in result:
            tool_name = entry.get("tool")
            query = entry.get("query")
            if tool_name and query:
                if tool_name in restricted_tools:
                    query = query[:1]
                state["selected_tools_queries"][tool_name] = query
                for tool in state['tool_list']:
                    if tool.name == tool_name:
                        state["tool_descriptions"].append(tool.description)
                        break
        if not state["selected_tools_queries"]:
            state["fallback"] = "clarification"
            state["fallback_response"] = fallback_response
        else:
            LOGGER.info("Valid tools prepared")
            LOGGER.info(state["selected_tools_queries"])
    except Exception as e:
        state["fallback"] = "clarification"
        state["fallback_response"] = fallback_response
        LOGGER.warning(e)
    return state


def run_retriever(state: MigraineQAState) -> MigraineQAState:
    start_time = time.time()
    if not state.get("selected_tools_queries"):
        state["fallback"] = "clarification"
        state["fallback_response"] = fallback_response
        LOGGER.warning("No Tools Available")
        return state

    state["context"] = []
    futures = []
    tool_map = {tool.name: tool for tool in state['tool_list']}

    with ThreadPoolExecutor() as executor:
        for tool_name, search_queries in state.get("selected_tools_queries", {}).items():
            tool = tool_map.get(tool_name)
            if not tool:
                continue
            for search_query in search_queries:
                future = executor.submit(_run_tool_query, tool, tool_name, search_query)
                future.tool_name = tool_name
                future.search_query = search_query
                futures.append(future)

        for future in as_completed(futures):
            try:
                docs = future.result()
                if docs:
                    state["context"].append(docs)
            except Exception as e:
                state["context"] = []
                state["fallback"] = "retriever_error"
                LOGGER.warning(f"retriever_error {e} (tool: {getattr(future, 'tool_name', 'unknown')}, query: {getattr(future, 'search_query', 'unknown')})")

    LOGGER.info("Context Build Successfully")
    LOGGER.info(f"Information Retrieval Time, {time.time() - start_time}")
    return state


def _run_tool_query(tool, tool_name, search_query):
    docs = tool.run(search_query)
    return f"Context From {tool_name}.\n{docs}"


def analyze_context(state: MigraineQAState) -> MigraineQAState:
    if not state.get("context"):
        state["attempts"] += 1
        if state["attempts"] < 1:
            state["fallback"] = "retry"
            LOGGER.info("Retry with tools")
        else:
            state["fallback"] = "retriever_error"
            state["fallback_response"] = fallback_response
            LOGGER.info("Retrial exhausted")
    return state


def generate_answer(state: MigraineQAState) -> MigraineQAState:
    start_time = time.time()
    context = "\n".join(state["context"]) if state["context"] else ""
    memory_context = "Previous conversation:\n" + "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in state["memory"]]) if state["memory"] else ""

    if context and memory_context:
        LOGGER.info("Generate answer with context and memory")
        prefix = "Use ONLY the following context and previous conversation"
    elif context:
        LOGGER.info("Generate answer with context only")
        prefix = "Use ONLY the following context"
    elif memory_context:
        LOGGER.info("Generate answer with memory only")
        prefix = "Use ONLY the following previous conversation"
    else:
        LOGGER.info("Generate answer from system prompt only")
        prefix = "Respond using only the system message"

    prompt = (
        f"{state['system_message_generate_answer']}.\n"
        f"{prefix} to answer the user's question.\n"
        "If you cannot answer, then respond politely and ask for clarification.\n"
        f"{memory_context}\n"
        f"User question: {state['question']}\n"
        f"Context:\n{context}"
    )
    response = state["openai_llm"].invoke([{"role": "user", "content": prompt}])
    state["answer"] = response.content
    LOGGER.info(f"Generate answer Time, {time.time() - start_time}")
    return state


def validate_answer(state: MigraineQAState) -> MigraineQAState:
    start_time = time.time()
    context = "\n".join(state["context"]) if state["context"] else "Not Available"
    memory_context = "Previous conversation:\n" + "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in state["memory"]]) if state["memory"] else "Not Available"

    validation_prompt = state["system_message_validator"].format(
        question=state["question"],
        answer=state["answer"],
        memory=memory_context,
        context=context,
        character=state["system_message_generate_answer"]
    )
    response = state["openai_llm"].invoke([{"role": "user", "content": validation_prompt}])
    result = response.content if isinstance(response.content, str) else "Invalid: LLM Error"

    state["answer_reasoning"] = result
    state["validated"] = True
    if "invalid" in result.lower():
        state["fallback"] = "incorrect_answer"
        state["fallback_response"] = fallback_invalid_response
        LOGGER.info("Validation failed")
    else:
        state["fallback"] = None
        state["fallback_response"] = ""
    LOGGER.info(f"Validate answer Time, {time.time() - start_time}")
    return state


def fallback_node(state: MigraineQAState) -> MigraineQAState:
    if state["fallback"] in ["clarification", "retriever_error", "incorrect_answer"]:
        state["answer"] = state["fallback_response"]
        LOGGER.warning("Fallback: setting fallback response")
    state["memory"].append({"role": "user", "content": state["question"]})
    state["memory"].append({"role": "assistant", "content": state["answer"]})
    LOGGER.warning("Fallback: memory updated")
    return state


def build_agentic_workflow():
    wf = StateGraph(MigraineQAState)
    wf.add_node("query_analyzer", query_analyzer)
    wf.add_node("run_retriever", run_retriever)
    wf.add_node("analyze_context", analyze_context)
    wf.add_node("generate_answer", generate_answer)
    wf.add_node("validate_answer", validate_answer)
    wf.add_node("fallback_node", fallback_node)

    wf.set_entry_point("query_analyzer")
    wf.add_edge("query_analyzer", "run_retriever")
    wf.add_edge("run_retriever", "analyze_context")
    wf.add_conditional_edges("analyze_context", lambda s: "query_analyzer" if s.get("fallback") == "retry" else "generate_answer")
    wf.add_edge("generate_answer", "validate_answer")
    wf.add_edge("validate_answer", "fallback_node")
    wf.add_edge("fallback_node", END)
    return wf.compile()


def build_initial_state(llm: AzureChatOpenAI, tool_list: List[Any], question: str = "") -> MigraineQAState:
    return MigraineQAState(
        question=question,
        system_message_generate_answer=system_message_generate_answer,
        system_message_query_analyzer=system_message_query_analyzer,
        system_message_validator=system_message_validator,
        selected_tools_queries={},
        tool_descriptions=None,
        context=[],
        answer=None,
        validated=None,
        answer_reasoning=None,
        attempts=0,
        fallback=None,
        fallback_response=None,
        memory=[],
        openai_llm=llm,
        tool_list=tool_list,
    )
