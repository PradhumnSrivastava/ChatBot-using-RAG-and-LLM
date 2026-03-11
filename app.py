import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

st.title("RAG Chatbot with Groq")
st.write("Upload a PDF and ask questions about its content.")

# Sidebar
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Groq model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDF
if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)

    st.session_state.vectorstore = vectorstore

    st.sidebar.success("PDF processed successfully.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
prompt = st.chat_input("Ask a question about the document")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.vectorstore is None:
        answer = "Please upload a PDF first."

    else:

        retriever = st.session_state.vectorstore.as_retriever()

        docs = retriever.invoke(prompt)

        context = "\n".join([doc.page_content for doc in docs])

        final_prompt = f"""
You are a research assistant.

Answer only using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{prompt}
"""

        with st.spinner("Generating response..."):

            response = llm.invoke(final_prompt)
            answer = response.content

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )