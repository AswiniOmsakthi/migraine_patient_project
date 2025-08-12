import streamlit as st

import sys

import os

 

# Add project root to Python path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(project_root)

 

from src.llm.multi_agents import (

    build_agentic_workflow,

    build_initial_state,

)

from src.knowledge_stores.migraine import load_vectorstore

from src.config import CHAT_MODEL_PARAMETERS

from langchain_openai import AzureChatOpenAI

 

# Initialize session state for chat history and memory

if 'chat_history' not in st.session_state:

    st.session_state.chat_history = []

if 'memory' not in st.session_state:

    st.session_state.memory = []

 

# Set page config

st.set_page_config(

    page_title="AI Chat Assistant",

    page_icon="💬",

    layout="wide"

)

 

st.title("AI Chat Assistant")

st.write("Ask me anything and I'll help you find the right information.")

 

# Streaming chat interface (chatbot style)

for role, message in st.session_state.chat_history:

    with st.chat_message("user" if role == "You" else "assistant"):

        st.markdown(message)

 

user_question = st.chat_input("Type your question...")

 

if user_question:

    st.session_state.chat_history.append(("You", user_question))

    with st.chat_message("user"):

        st.markdown(user_question)

 

    # Build LLM and workflow

    llm = AzureChatOpenAI(

        azure_deployment=CHAT_MODEL_PARAMETERS["engine"],

        api_version="2023-05-15",

        temperature=CHAT_MODEL_PARAMETERS["temperature"],

        max_tokens=CHAT_MODEL_PARAMETERS["max_tokens"],

        top_p=CHAT_MODEL_PARAMETERS["top_p"],

        frequency_penalty=CHAT_MODEL_PARAMETERS["frequency_penalty"],

        presence_penalty=CHAT_MODEL_PARAMETERS["presence_penalty"],

        timeout=120,

    )

    tool_list = load_vectorstore(None)

    workflow = build_agentic_workflow()

    state = build_initial_state(llm, tool_list, question=user_question)

    state["memory"] = st.session_state.memory

 

    # Run workflow

    result = workflow.invoke(state)

    answer = result.get("answer", "Sorry, I couldn't generate an answer.")

 

    # Save memory for next turn

    st.session_state.memory = result.get("memory", st.session_state.memory)

    st.session_state.chat_history.append(("Assistant", answer))

    with st.chat_message("assistant"):

        st.markdown(answer)

 

# Add clear chat button at the bottom

if st.button("Clear Chat History"):

    st.session_state.chat_history = []

    st.session_state.memory = []

    st.rerun()