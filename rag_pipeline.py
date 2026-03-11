from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def rag_answer(question):

    docs = retriever.get_relevant_documents(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the context to answer the question.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content