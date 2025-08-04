# test_agentic_workflow.py

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from src.llm.multi_agents import build_initial_state, build_agentic_workflow

# Load Azure credentials from .env
load_dotenv()

# Set up Azure OpenAI LLM client
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_MODEL_DEPLOYEMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Dummy Tool for testing
class DummyTool:
    name = "migraine_qa_csv"
    description = "CSV tool to answer migraine-related queries."

    def run(self, query):
        return f"CSV result for query: {query}"

# Build tool list and initial state
tool_list = [DummyTool()]
question = "How effective is Aimovig in treating chronic migraine?"
state = build_initial_state(llm=llm, tool_list=tool_list, question=question)

# Build graph and invoke it
graph = build_agentic_workflow()
final_state = graph.invoke(state)

# Display final results
print("\n🎯 Final Answer:")
print(final_state["answer"])
print("\n📜 Reasoning:")
print(final_state["answer_reasoning"])
print("\n🧠 Memory:")
print(final_state["memory"])
print("\n✅ Validated:", final_state["validated"])