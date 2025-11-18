import streamlit as st
from datetime import datetime
from langchain_classic.memory import ConversationBufferMemory
from chatbot_backend import (
    fetch_transcript, 
    generate_summary, 
    build_vector_store, 
    RAG_pipeline,
    get_source_documents
)

st.set_page_config(
    page_title="YouTube RAG Assistant", 
    layout="wide",
    page_icon="🎥"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "video_id" not in st.session_state:
    st.session_state["video_id"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "transcript" not in st.session_state:
    st.session_state["transcript"] = ""
if "transcript_chunks" not in st.session_state:
    st.session_state["transcript_chunks"] = []
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "chain" not in st.session_state:
    st.session_state["chain"] = None
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
if "show_sources" not in st.session_state:
    st.session_state["show_sources"] = True
if "error_message" not in st.session_state:
    st.session_state["error_message"] = ""
if "usage_stats" not in st.session_state:
    st.session_state["usage_stats"] = {"queries": 0}


@st.cache_data
def cached_fetch_transcript(vid):
    """Cache transcript fetching to avoid redundant API calls."""
    return fetch_transcript(vid)


@st.cache_resource
def cached_vector_store(transcript, _transcript_chunks):
    """Cache vector store building."""
    return build_vector_store(transcript, _transcript_chunks)


def export_chat():
    """Export chat history as text."""
    if not st.session_state["messages"]:
        return "No chat history to export."
    
    export_text = f"YouTube Video Chat Export\n"
    export_text += f"Video ID: {st.session_state['video_id']}\n"
    export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += "="*50 + "\n\n"
    
    for msg in st.session_state["messages"]:
        role = msg["role"].upper()
        content = msg["content"]
        export_text += f"{role}:\n{content}\n\n"
    
    return export_text


def format_timestamp(seconds):
    """Convert seconds to MM:SS format."""
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}:{secs:02d}"


# Main UI Layout
st.title("YouTube Video Assistant with RAG")

col1, col2 = st.columns([1, 2], gap="large")

# Left Column - Video Input & Info
with col1:
    st.subheader("Video Input")
    
    vid = st.text_input(
        "YouTube Video ID", 
        placeholder="e.g., dQw4w9WgXcQ",
        help="Enter the video ID from the YouTube URL"
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        btn_process = st.button("Process Video", type="primary")
    with col_btn2:
        btn_clear = st.button("Clear Chat")
    
    # Settings
    with st.expander("⚙️"):
        st.session_state["show_sources"] = st.checkbox(
            "Show source timestamps", 
            value=st.session_state["show_sources"]
        )
    
    # Display video
    if vid:
        st.session_state["video_id"] = vid
        vid_url = f"https://www.youtube.com/watch?v={vid}"
        st.video(vid_url)
    
    # Process button logic
    if btn_process:
        if not vid:
            st.error("Please enter a YouTube video ID")
        else:
            with st.spinner("Fetching transcript..."):
                transcript, chunks, error = cached_fetch_transcript(vid)
                
                if error:
                    st.error(f"❌ {error}")
                    st.session_state["error_message"] = error
                else:
                    st.session_state["transcript"] = transcript
                    st.session_state["transcript_chunks"] = chunks
                    st.session_state["error_message"] = ""
                    st.success("Transcript fetched successfully!")
            
            if st.session_state["transcript"]:
                with st.spinner("Generating summary..."):
                    summary, stats = generate_summary(st.session_state["transcript"])
                    st.session_state["summary"] = summary
                    st.session_state["usage_stats"]["summary_generated"] = True
                
                with st.spinner("Building knowledge base..."):
                    st.session_state["vector_store"] = cached_vector_store(
                        st.session_state["transcript"],
                        st.session_state["transcript_chunks"]
                    )
                    st.session_state["chain"] = RAG_pipeline(
                        st.session_state["vector_store"],
                        st.session_state["memory"]
                    )
                    st.success("Ready to answer questions!")
    
    # Clear chat logic
    if btn_clear:
        st.session_state["messages"] = []
        st.session_state["memory"].clear()
        st.session_state["usage_stats"]["queries"] = 0
        st.rerun()
    
    # Display summary
    if st.session_state["summary"]:
        st.markdown("---")
        st.markdown("### Video Summary")
        st.info(st.session_state["summary"])


# Right Column - Chat Interface
with col2:
    st.subheader("Chat with Video")
    
    # Export button
    if st.session_state["messages"]:
        export_text = export_chat()
        st.download_button(
            label="Export Chat",
            data=export_text,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Chat container
    chat_container = st.container(height=500)
    
    # Display messages
    with chat_container:
        if not st.session_state["messages"]:
            st.info("Ask questions about the video after processing it!")
        
        for i, message in enumerate(st.session_state["messages"]):
            role = message.get("role", "assistant")
            content = message.get("content", "")
            
            with st.chat_message(role):
                st.markdown(content)
                
                # Show sources if available and enabled
                if (role == "assistant" and 
                    st.session_state["show_sources"] and 
                    "sources" in message):
                    with st.expander("View Sources"):
                        for j, source in enumerate(message["sources"], 1):
                            timestamp = source["timestamp"]
                            if timestamp != "N/A":
                                time_str = format_timestamp(timestamp)
                                st.markdown(
                                    f"**Source {j}** @ [{time_str}]"
                                    f"(https://youtube.com/watch?v={st.session_state['video_id']}&t={int(timestamp)}s)"
                                )
                            st.markdown(f"> {source['content']}")
                            st.markdown("---")
    
    # Chat input
    prompt = st.chat_input(
        "Ask a question about the video...",
        disabled=not st.session_state["vector_store"]
    )
    
    # Process user input
    if prompt:
        if not st.session_state["vector_store"]:
            st.warning("Please process a video first!")
        else:
            # Add user message
            st.session_state["messages"].append({
                "role": "user", 
                "content": prompt
            })
            
            # Get response
            with st.spinner("Thinking..."):
                try:
                    # Get AI response
                    ai_response = st.session_state["chain"].invoke({
                        "question": prompt
                    })
                    
                    # Get source documents
                    sources = get_source_documents(
                        st.session_state["vector_store"], 
                        prompt
                    )
                    
                    # Update memory
                    st.session_state["memory"].save_context(
                        {"input": prompt},
                        {"output": ai_response}
                    )
                    
                    # Add assistant message
                    st.session_state["messages"].append({
                        "role": "assistant", 
                        "content": ai_response,
                        "sources": sources
                    })
                    
                    # Update stats
                    st.session_state["usage_stats"]["queries"] += 1
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })
            
            st.rerun()

# Footer
st.markdown("---")
st.caption("Built with Streamlit, LangChain, and OpenAI | RAG-powered Q&A System")