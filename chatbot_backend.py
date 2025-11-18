from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import os

load_dotenv()


def format_context_text(retrieved_docs: List[Document]) -> str:
    """Format retrieved documents into context string with source timestamps."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        timestamp = doc.metadata.get('timestamp', 'N/A')
        context_parts.append(f"[Source {i} @ {timestamp}s]:\n{doc.page_content}")
    return "\n\n".join(context_parts)


def fetch_transcript(video_id: str) -> Tuple[str, List[Dict], str]:
    """
    Fetch YouTube transcript with timestamp information.
    
    Returns:
        Tuple of (transcript_text, transcript_chunks, error_message)
    """
    try:
        # Create API instance and fetch transcript
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        
        # Convert to raw data format
        transcript_data = fetched_transcript.to_raw_data()
        
        # Keep chunks with timestamps for better retrieval
        transcript_chunks = []
        full_text_parts = []
        
        for chunk in transcript_data:
            transcript_chunks.append({
                'text': chunk['text'],
                'start': chunk['start'],
                'duration': chunk.get('duration', 0)
            })
            full_text_parts.append(chunk['text'])
        
        transcript_text = " ".join(full_text_parts)
        return transcript_text, transcript_chunks, ""
        
    except TranscriptsDisabled:
        return "", [], "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "", [], "No transcript available for this video."
    except CouldNotRetrieveTranscript:
        return "", [], "Could not retrieve transcript due to API restrictions."
    except Exception as e:
        return "", [], f"Unexpected error: {str(e)}"


def generate_summary(transcript: str) -> Tuple[str, Dict]:
    """
    Generate summary with token usage tracking.
    
    Returns:
        Tuple of (summary_text, usage_stats)
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    parser = StrOutputParser()
    
    prompt = PromptTemplate(
        template="""Analyze this YouTube video transcript and provide:
1. A brief 2-3 sentence summary
2. 3-5 key topics discussed
3. Main takeaway

Transcript: {transcript}

Format your response clearly with sections.""",
        input_variables=["transcript"]
    )
    
    chain = prompt | llm | parser
    
    # For tracking, we'll estimate tokens (more accurate tracking would use callbacks)
    summary = chain.invoke({"transcript": transcript[:4000]})  # Limit for cost control
    
    usage_stats = {
        "summary_generated": True,
        "transcript_length": len(transcript)
    }
    
    return summary, usage_stats


def build_vector_store(transcript: str, transcript_chunks: List[Dict]) -> FAISS:
    """
    Build vector store with timestamp metadata for source attribution.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create documents with timestamp metadata
    documents = []
    current_pos = 0
    
    for chunk in transcript_chunks:
        chunk_text = chunk['text']
        chunk_start = chunk['start']
        
        # Find this chunk in the full transcript
        pos = transcript.find(chunk_text, current_pos)
        if pos != -1:
            current_pos = pos + len(chunk_text)
        
        # Create document with metadata
        doc = Document(
            page_content=chunk_text,
            metadata={"timestamp": int(chunk_start)}
        )
        documents.append(doc)
    
    # Split documents
    split_docs = splitter.split_documents(documents)
    
    # Build vector store
    vector_store = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    
    return vector_store


def RAG_pipeline(vector_store: FAISS, memory: ConversationBufferMemory) -> any:
    """
    Create RAG pipeline with memory and source attribution.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 chunks
    )
    
    parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = PromptTemplate(
        template="""You are a helpful AI assistant for YouTube video analysis.
            Answer the question based ONLY on the provided transcript context.
            If the context doesn't contain relevant information, say "I don't have enough information from the video to answer that."

            Transcript Context:
            {context}

            Previous Conversation:
            {chat_history}

            Current Question: {question}

            Provide a clear, concise answer. If referencing specific parts, mention the timestamp.""",
        input_variables=["context", "chat_history", "question"]
    )
    
    def get_chat_history(_):
        """Retrieve chat history from memory."""
        return memory.load_memory_variables({}).get("history", "")
    
    chain = (
        RunnableParallel({
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(get_chat_history),
            "context": (
                RunnableLambda(lambda inp: inp["question"]) 
                | retriever 
                | RunnableLambda(format_context_text)
            )
        })
        | prompt 
        | llm 
        | parser
    )
    
    return chain


def get_source_documents(vector_store: FAISS, query: str, k: int = 3) -> List[Dict]:
    """
    Retrieve source documents for a query with metadata.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    
    sources = []
    for doc in docs:
        sources.append({
            "content": doc.page_content[:200] + "...",  # Preview
            "timestamp": doc.metadata.get("timestamp", "N/A")
        })
    
    return sources