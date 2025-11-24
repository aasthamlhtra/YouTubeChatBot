# YouTube Video Assistant with RAG

A powerful Retrieval-Augmented Generation (RAG) chatbot that allows you to have intelligent conversations about YouTube videos.  
The assistant fetches video transcripts, generates summaries, and answers questions based on the video content with source attribution and timestamps.

## Features

- Automatic Transcript Fetching: Retrieves YouTube video transcripts using the video ID.
- AI-Powered Summarization: Generates concise summaries with key topics and takeaways.
- RAG-Based Q&A: Ask questions about the video content and get accurate, context-aware answers.
- Source Attribution: Every answer includes timestamps linking back to relevant video segments.
- Conversation Memory: Maintains chat history for contextual follow-up questions.
- Export Functionality: Download your chat history for later reference.
- Interactive UI: Clean, user-friendly Streamlit interface with embedded video player.

## Architecture

The project uses a RAG pipeline with the following components:

- Transcript Retrieval: Fetches and processes YouTube video transcripts with timestamps.
- Vector Store: Builds a FAISS vector database with OpenAI embeddings for semantic search.
- LangChain Pipeline: Implements a RAG chain with conversation memory.
- Streamlit Interface: Provides an intuitive web UI for interaction.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Internet connection for YouTube transcript fetching

## Installation

### Clone the repository:

```bash
git clone <repo-url>
cd youtube-rag-assistant
```

### Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Create a .env file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Start the Streamlit application:

```bash
streamlit run chatbot_ui.py
```

Then:

- Open your browser at http://localhost:8501  
- Enter a YouTube video ID (e.g., dQw4w9WgXcQ)  
- Click "Process Video" to fetch the transcript and build the knowledge base  
- Start asking questions about the video in the chat interface  

## Project Structure

```
youtube-rag-assistant/
│
├── chatbot_backend.py     # Core RAG logic and pipeline
├── chatbot_ui.py          # Streamlit user interface
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (create this)
└── README.md              # This file
```

## Key Functions

### Backend (chatbot_backend.py)

- fetch_transcript(video_id): Retrieves YouTube transcript with timestamps  
- generate_summary(transcript): Creates AI-powered video summary  
- build_vector_store(transcript, chunks): Builds FAISS vector database  
- RAG_pipeline(vector_store, memory): Creates the RAG chain with memory  
- get_source_documents(vector_store, query): Retrieves relevant source chunks  

### Frontend (chatbot_ui.py)

- Interactive video player  
- Real-time chat interface  
- Source timestamp display  
- Chat history export  
- Session state management  

## Configuration

The application uses the following default settings:

| Setting | Value |
|--------|-------|
| LLM Model | GPT-3.5-turbo |
| Chunk Size | 1000 characters |
| Chunk Overlap | 200 characters |
| Retrieved Documents | Top 3 most relevant chunks |
| Temperature | 0.7 for chat, 0.3 for summaries |

You can modify these settings in chatbot_backend.py.

## Features in Detail

### Source Attribution
Every assistant response includes clickable timestamps that link directly to the relevant portions of the YouTube video, making verification easy.

### Conversation Memory
The chatbot maintains context across the conversation, allowing for natural follow-up questions without repeating information.

### Error Handling
Graceful handling of scenarios including:

- Videos with disabled transcripts  
- Missing transcripts  
- API errors  
- Invalid video IDs  

## Limitations

- Only works with YouTube videos that have available transcripts  
- Requires an active OpenAI API key (usage incurs costs)  
- Transcript quality depends on YouTube's auto-generated captions  
- Large videos may take longer to process  

## Troubleshooting

**Issue:** "Transcripts are disabled for this video"  
**Solution:** The video owner has disabled transcripts. Try a different video.

**Issue:** OpenAI API errors  
**Solution:** Check your API key in the .env file and ensure you have sufficient credits.

**Issue:** Installation errors  
**Solution:** Ensure you're using Python 3.8+ and all dependencies are correctly installed.

## Future Enhancements

- Support for multiple video comparisons  
- Multi-language transcript support  
- Advanced search filters  
- Video playlist analysis  
- Custom embedding models  
- Export to PDF format  

## Dependencies

- streamlit  
- langchain  
- openai  
- faiss-cpu  
- youtube-transcript-api  
- python-dotenv  

See `requirements.txt` for complete list with versions.

