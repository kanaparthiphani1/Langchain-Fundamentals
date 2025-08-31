import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import time

# Load env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# LLM
model = ChatGroq(model="Gemma2-9b-It", api_key=groq_api_key)

# Prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that remembers the conversation."),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

# Runnable chain
chain = prompt | model

# Store histories in session_state
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Wrap with memory
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# UI
st.title("💬 Chat with Groq")

session_id = "chat1"
history = get_session_history(session_id)

# Display existing history in chat format
for msg in history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input at bottom
if prompt := st.chat_input("Ask me anything..."):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    response = with_history.invoke(
        {"question": prompt},
        config={"configurable": {"session_id": session_id}},
    )

    # Show assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for chunk in response.content.split():
            full_text += chunk + " "
            placeholder.markdown(full_text + "▌")  # ▌ acts like a cursor
            time.sleep(0.05)  # typing speed
        placeholder.markdown(full_text)  # final text without cursor
