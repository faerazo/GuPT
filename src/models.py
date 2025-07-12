from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

class RouteQuery(BaseModel):
    """Route a user query to the most relevant content type."""
    content_type: Literal["course", "program", "both"] = Field(
        ...,
        description="Route to: 'course' for specific course questions, 'program' for program questions, 'both' when the question involves both or is unclear"
    )

@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""
    source: str
    type: str  # 'markdown' or 'pdf'
    category: str  # 'courses' or 'programs'
    doc_type: str  # 'course' or 'program'
    filename: str
    course_code: Optional[str] = None

@dataclass
class QueryResult:
    """Result of a RAG query."""
    answer: str
    source_documents: List[Any]  # List of Document objects
    content_type: str
    processing_time: Optional[float] = None
    generated_queries: Optional[List[str]] = None
    retrieval_stats: Optional[Dict[str, Any]] = None

@dataclass
class ChatInteraction:
    """Single chat interaction for logging."""
    timestamp: str
    query: Dict[str, Any]
    retrieval: Dict[str, Any]
    response: Dict[str, str]
    performance: Dict[str, Any]
    chat_context: Dict[str, Any]
    system_info: Dict[str, Any]

@dataclass
class RetrievalStats:
    """Statistics about document retrieval."""
    total_documents: int
    document_types: Dict[str, int]
    search_config: Dict[str, Any]
    queries_used: List[str]

class EmbeddingConfig(BaseModel):
    """Configuration for embeddings."""
    model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    max_retries: int = 3
    request_timeout: int = 60

class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: Optional[int] = None

class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    persist_directory: str = "./data/chroma"
    collection_name: str = "course_docs"
    collection_metadata: Dict[str, str] = Field(default_factory=lambda: {"hnsw:space": "cosine"})

class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""
    search_type: str = "mmr"
    k_values: Dict[str, int] = Field(default_factory=lambda: {
        "course": 6,
        "program": 15,
        "both": 15
    })
    fetch_k_multiplier: int = 3

@dataclass
class ProcessingStats:
    """Statistics about document processing."""
    total_documents: int
    courses_processed: int
    programs_processed: int
    chunks_created: int
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class ChatMemoryMessage(BaseModel):
    """Message in chat memory."""
    role: str
    content: str
    timestamp: Optional[str] = None

class SystemStatus(BaseModel):
    """System status information."""
    database_initialized: bool = False
    documents_loaded: int = 0
    model_version: str = ""
    embedding_version: str = ""
    last_updated: Optional[str] = None 