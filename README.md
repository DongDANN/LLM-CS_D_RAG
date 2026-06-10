# 🦷 Dental AI Assistant

A RAG-powered dental clinic chatbot built with LangChain, Google Gemini, and Streamlit.

## Architecture

```
User Question → Streamlit UI → LangChain Chain
                                      ↓
                              FAISS Retriever (top 3 chunks)
                                      ↓
                              Gemini 2.0 Flash + Context → Answer
```

**RAG Pipeline:**
1. Dental knowledge documents (Markdown/PDF) are loaded from `data/`
2. Documents are split into 500-character chunks with 50-char overlap
3. Chunks are embedded using HuggingFace `all-MiniLM-L6-v2`
4. Embeddings are stored in a FAISS vector store
5. At query time, the 3 most relevant chunks are retrieved and passed to Gemini

## Setup

### 1. Get a Google API Key (free)
- Go to [Google AI Studio](https://aistudio.google.com/apikey)
- Create an API key
- Copy `.env.example` to `.env` and paste your key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `GOOGLE_API_KEY` in the Streamlit secrets dashboard
5. Deploy

## Project Structure

```
├── app.py                  # Streamlit chat interface
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── data/
│   └── dental_knowledge.md # Clinic knowledge base
├── src/
│   ├── rag_pipeline.py     # Document loading, chunking, embedding, retrieval
│   └── chain.py            # LangChain + Gemini chain setup
└── vectorstore/            # Auto-generated FAISS index
```

## Adding Your Own Data

Drop any `.md` or `.pdf` files into the `data/` folder and restart the app. The vector store will rebuild automatically.
