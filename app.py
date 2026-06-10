import streamlit as st
from src.chain import get_chain

st.set_page_config(
    page_title="Dental AI Assistant",
    page_icon=None,
    layout="centered",
)

st.title("Dental AI Assistant")
st.caption("Ask me anything about dental services, procedures, pricing, and aftercare.")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This AI assistant helps answer questions about dental clinic services
        using a **RAG (Retrieval-Augmented Generation)** pipeline.

        ---

        **Developer:**
        Ronald Sullano - DONGDANN

        ---

        **How It Works:**

        1. **You ask a question** in the chat input below.
        2. **Document Retrieval** - Your question is converted into
           a vector embedding using Google's Gemini Embedding model.
           The system searches the FAISS vector store for the 3 most
           relevant text chunks from the dental knowledge base.
        3. **Context Injection** - The retrieved chunks are injected
           into a prompt template as context for the LLM.
        4. **LLM Generation** - Google Gemini 2.5 Flash reads the
           context + your question and generates a grounded answer.
        5. **Response** - The answer is displayed in the chat.

        ---

        **Tech Stack:**
        - **LLM:** Google Gemini 2.5 Flash
        - **Embeddings:** Google Gemini Embedding
        - **Vector Store:** FAISS (local, in-memory)
        - **Orchestration:** LangChain (LCEL)
        - **UI:** Streamlit
        - **Data:** Markdown documents in /data folder

        ---

        **RAG Pipeline Architecture:**
        ```
        User Question
             |
             v
        Embedding (Google)
             |
             v
        FAISS Similarity Search
             |
             v
        Top 3 Relevant Chunks
             |
             v
        Prompt Template + Context
             |
             v
        Gemini 2.5 Flash LLM
             |
             v
        Generated Answer
        ```

        ---

        **Note:** Pricing is in Philippine Pesos and are estimates only.
        """
    )
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource
def load_chain():
    return get_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a dental question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = load_chain()
            response = chain.invoke(prompt)

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
