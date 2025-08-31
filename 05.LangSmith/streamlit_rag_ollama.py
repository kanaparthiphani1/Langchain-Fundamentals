import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama

# ------------------ Load Environment ------------------
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# ------------------ Config ------------------
DB_PATH = "faiss_index"

# ------------------ Prepare Retriever ------------------
@st.cache_resource(show_spinner="Preparing documents and embeddings (first run may take time)...")
def prepare_retriever():
    if os.path.exists(DB_PATH):
        # Load existing FAISS index
        vectordb = FAISS.load_local(DB_PATH, OllamaEmbeddings(model="gemma:2b") ,allow_dangerous_deserialization=True)
    else:
        # Load and process document
        loader = WebBaseLoader("https://hellointerview.substack.com/p/design-a-rate-limiter")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        # Use a fast embedding model for vector DB
        embedding = OllamaEmbeddings(model="gemma:2b")
        vectordb = FAISS.from_documents(split_docs, embedding)

        # Save FAISS for reuse
        vectordb.save_local(DB_PATH)

    return vectordb.as_retriever()

retriever = prepare_retriever()

# ------------------ Streamlit UI ------------------
st.title("LangChain Demo with Gemma Model (Rate Limiter Article)")
st.write("Ask me anything about Rate Limiter (from the provided article).")

# Text input (always visible)
input_text = st.text_input("What question do you have in mind?")

# ------------------ Answer Generation ------------------
if input_text:
    with st.spinner("Thinking..."):
        # Use Gemma:2b as the LLM (for answers)
        llm = Ollama(model="gemma:2b")

        # Prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        result = retriever_chain.invoke({"input": input_text})

        st.subheader("Answer:")
        st.write(result["answer"])
