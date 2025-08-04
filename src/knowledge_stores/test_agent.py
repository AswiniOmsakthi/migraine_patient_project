# test_agent_simple_fixed.py
import os
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import Tool, BaseTool
from langchain.agents import initialize_agent, AgentType
from src.knowledge_stores.migraine import load_vectorstore

def ask(question: str) -> str:
    tools = load_vectorstore()

    print("DEBUG: Inspecting tool schemas...")
    for t in tools:
        if hasattr(t, "get_input_schema"):
            schema_model = t.get_input_schema()
            props = schema_model.schema()["properties"]
        else:
            # Fallback: args_schema
            schema_model = t.args_schema
            props = getattr(schema_model, "schema", lambda: {"properties":{}})()["properties"]
        keys = list(props.keys())
        print(f"  • {t.name}: accepts {keys}")

    wrapped = []
    for t in tools:
        def make_wrapper(orig_tool: BaseTool):
            def wrapper(input_str: str):
                schema_model = orig_tool.get_input_schema()
                props = schema_model.schema().get("properties", {})
                data = {}
                if input_str and props:
                    first_arg = next(iter(props.keys()))
                    data[first_arg] = input_str
                else:
                    # Fall back to 'query' if it's present
                    if "query" in props:
                        data["query"] = input_str
                return orig_tool.invoke(data)
            return wrapper

        wrapped.append(
            Tool(
                name=t.name,
                func=make_wrapper(t),
                description=getattr(t, "description", ""),
                args_schema=type("A", (), {"schema": lambda self: {"properties": {}}})  # dummy
            )
        )

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT"),
        temperature=0,
    )

    agent = initialize_agent(
        tools=wrapped,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=True,
    )

    return agent.run(question)

if __name__ == "__main__":
    print(ask("What brand names does erenumab go by?"))