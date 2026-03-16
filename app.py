import streamlit as st
from assistant import EduTubeAssistant

# RAG paths
RAG_DB_DIR = "ragdb/chromadb_store2"
RAG_CLEAN_DIR = "ragdb/cleandata2"

# Page configuration
st.set_page_config(
    page_title="EduTube AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 EduTube AI Assistant")
st.caption("Ask questions about programming concepts from educational videos")

# Initialize assistant (cached)
@st.cache_resource
def load_assistant():
    try:
        return EduTubeAssistant(
            model_name="edutube-llama",
            db_dir=RAG_DB_DIR,
            clean_dir=RAG_CLEAN_DIR
        )
    except Exception as e:
        st.error(f"Failed to load assistant: {str(e)}")
        return None

assistant = load_assistant()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    model = st.selectbox(
        "Select Model",
        ["edutube-llama", "llama3.2"],
        index=0
    )

    if assistant and model != assistant.model_name:
        assistant.model_name = model

    st.subheader("🔍 RAG Settings")
    use_rag = st.checkbox("Use RAG Context", value=True, disabled=assistant is None)
    n_results = st.slider("Context Chunks", 1, 10, 5)

    st.subheader("🤖 Model Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 4000, 2000, 100)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}. {source['video_title']}** (Relevance: {source['score']})")
                    if source.get("video_url"):
                        st.caption(f"🔗 Watch video")

if prompt := st.chat_input("Ask about programming concepts..."):
    if not assistant:
        st.error("❌ Assistant not available. Please check the configuration.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []

        try:
            if use_rag:
                with st.spinner("🔍 Searching knowledge base..."):
                    pass

                for event in assistant.query_with_context_stream(
                    question=prompt,
                    n_results=n_results,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    if event["type"] == "context":
                        sources = event["sources"]
                        if sources:
                            with st.expander("📚 Retrieved Context", expanded=False):
                                for i, hit in enumerate(sources, 1):
                                    st.markdown(f"**{i}. {hit['video_title']}** (Relevance: {hit['score']})")
                                    st.text(hit["text"][:300] + "...")
                                    st.divider()

                    elif event["type"] == "chunk":
                        full_response += event["text"]
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                if sources:
                    with st.expander("📚 Sources Used"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**{i}. {source['video_title']}** (Relevance: {source['score']})")
                            if source.get("video_url"):
                                st.caption(f"🔗 Watch video")
                            st.divider()

            else:
                import ollama
                stream = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                )

                for chunk in stream:
                    if chunk["message"]["content"]:
                        full_response += chunk["message"]["content"]
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            st.error(error_message)
            full_response = error_message

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources if use_rag else []
    })