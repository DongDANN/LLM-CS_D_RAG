import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"


def load_documents():
    documents = []

    md_loader = DirectoryLoader(
        str(DATA_DIR), glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents.extend(md_loader.load())

    pdf_loader = DirectoryLoader(
        str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader,
    )
    documents.extend(pdf_loader.load())

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def build_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore


def load_vectorstore(embeddings):
    if VECTORSTORE_DIR.exists():
        return FAISS.load_local(
            str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
        )
    return None


def get_retriever():
    embeddings = get_embeddings()

    vectorstore = load_vectorstore(embeddings)
    if vectorstore is None:
        documents = load_documents()
        chunks = chunk_documents(documents)
        vectorstore = build_vectorstore(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})
