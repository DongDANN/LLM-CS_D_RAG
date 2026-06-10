import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.rag_pipeline import get_retriever

load_dotenv()

SYSTEM_PROMPT = """You are a friendly and knowledgeable dental clinic assistant.
Your role is to help patients with questions about dental services, procedures,
pricing, aftercare, and appointments.

Rules:
- Only answer questions related to dental health and our clinic services.
- If the question is not about dental topics, politely say you can only help with dental-related inquiries.
- Base your answers on the provided context. If the context doesn't contain the answer, say so honestly.
- Be warm, professional, and reassuring — many patients are anxious about dental visits.
- When discussing prices, mention that these are estimates and may vary.

Context from our clinic knowledge base:
{context}

Patient's question: {question}

Helpful answer:"""

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Set it in your .env file.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )

    retriever = get_retriever()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
