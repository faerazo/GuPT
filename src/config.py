import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration settings."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini-2025-04-14")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    
    # Database Configuration
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "course_docs")
    
    # Text Splitting Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    RETRIEVAL_K_VALUES = {
        "course": int(os.getenv("RETRIEVAL_K_COURSE", "20")),
        "program": int(os.getenv("RETRIEVAL_K_PROGRAM", "15")),
        "both": int(os.getenv("RETRIEVAL_K_BOTH", "25"))
    }
    
    # Embedding Configuration
    EMBEDDING_CHUNK_SIZE = int(os.getenv("EMBEDDING_CHUNK_SIZE", "1000"))
    EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
    EMBEDDING_REQUEST_TIMEOUT = int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "60"))
    
    # Memory Configuration
    MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "3"))
    
    # Logging Configuration
    LOG_FILE = os.getenv("LOG_FILE", "chat_history.json")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # Directory Paths
    DATA_BASE_PATH = os.getenv("DATA_BASE_PATH", "./data")
    COURSES_MD_PATH = os.getenv("COURSES_MD_PATH", "data/courses/md")
    COURSES_PDF_PATH = os.getenv("COURSES_PDF_PATH", "data/courses/pdf")
    PROGRAMS_MD_PATH = os.getenv("PROGRAMS_MD_PATH", "data/programs/md")
    PROGRAMS_PDF_PATH = os.getenv("PROGRAMS_PDF_PATH", "data/programs/pdf")
    
    # Interface Configuration
    GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    # Telemetry Configuration
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    ANONYMIZED_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "false").lower() == "true"
    POSTHOG_DISABLED = os.getenv("POSTHOG_DISABLED", "true").lower() == "true"
    CHROMA_TELEMETRY_DISABLED = os.getenv("CHROMA_TELEMETRY_DISABLED", "true").lower() == "true"
    DO_NOT_TRACK = os.getenv("DO_NOT_TRACK", "1")

class PromptTemplates:
    """Centralized prompt templates."""
    
    COURSE_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents about university COURSES. 

Follow these guidelines:
1. Focus on different aspects: content, prerequisites, learning outcomes, examination methods
2. Use different phrasings and synonyms
3. Include the course code or name if present in the original question
4. Make queries both more specific and more general than the original
5. Ensure each query is semantically meaningful and complete

Original question: {question}

Generate 5 different versions, one per line."""

    PROGRAM_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents about university PROGRAMS. 

Follow these guidelines:
1. Focus on different aspects: program structure, career opportunities, admission requirements, outcomes
2. Use different phrasings and synonyms
3. Include the program name if present in the original question
4. Make queries both more specific and more general than the original
5. Consider both overall program information and specific details

Original question: {question}

Generate 5 different versions, one per line."""

    GENERAL_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents about both university COURSES and PROGRAMS. 

Follow these guidelines:
1. Balance between course-specific and program-level information
2. Include variations that focus on how courses fit into programs
3. Use different phrasings and synonyms
4. Make queries both more specific and more general than the original
5. Maintain the original intent while exploring different aspects

Original question: {question}

Generate 5 different versions, one per line."""

    ROUTER_SYSTEM_TEMPLATE = """You are an expert at routing user questions about university education to the appropriate content type.
Your task is to determine whether the question is about:
1. A specific COURSE or course-related information
2. A specific PROGRAM or program-related information
3. BOTH when the question involves both courses and programs or when it's unclear

Examples:
- "What are the prerequisites for DIT134?" -> course
- "Tell me about the Software Engineering program" -> program
- "What courses are included in the Data Science master's?" -> both
- "How many credits do I need?" -> both"""

    SYSTEM_TEMPLATE = """You are a helpful course and program information assistant for Gothenburg University. 
Your role is to provide accurate information about courses and programs based ONLY on the provided course and program documents.

Important rules to follow:
1. Only answer questions about courses that are explicitly mentioned in the provided documents
2. If a course is not in the documents, clearly state that you don't have information about that course
3. Base your answers solely on the content from the course documents
4. If you're unsure about any information, say so explicitly
5. When discussing course content, prerequisites, or evaluation methods, quote directly from the source documents when possible
6. Include the course code (e.g., DIT123) when referring to courses
7. For listing questions (e.g., "What programs are available?", "List all courses in X"):
   - ALWAYS check the sources list first
   - THOROUGHLY examine EACH source document listed in the sources
   - List EVERY program or course mentioned in ANY of the retrieved documents
   - Do not skip any programs even if they seem similar to others
   - Include program/course codes when available
   - Group items logically (e.g., by degree level: Bachelor's, Master's)
   - Double-check the sources list against your response to ensure no programs were missed
8. For questions asking about all programs from a specific school/department:
   - List ALL programs from the retrieved documents
   - Cross-reference the sources list with your response to ensure completeness
   - Include full program names and codes
   - Organize by degree level (Bachelor's/Master's)
   - Specify the credit amount if available
   - Before finishing your response, verify that you've included every program from every source listed

Context from documents: {context}

Current conversation history: {chat_history}

Human question: {question}

Remember: 
1. When asked to list programs or courses, THOROUGHLY check all retrieved documents and include EVERY relevant item. 
2. Do not summarize or skip any programs/courses found in the sources.
3. Always cross-reference your final list against the sources to ensure nothing was missed.
4. If you see a source in the list that contains "programme" or "program" in its name, make sure that program is included in your response.

Please provide a response based strictly on the above context. If the information isn't in the context, say so."""

    @classmethod
    def get_query_template(cls, content_type: str) -> str:
        """Get the appropriate query template based on content type."""
        templates = {
            "course": cls.COURSE_QUERY_TEMPLATE,
            "program": cls.PROGRAM_QUERY_TEMPLATE,
            "both": cls.GENERAL_QUERY_TEMPLATE
        }
        return templates.get(content_type, cls.GENERAL_QUERY_TEMPLATE)

class AppConstants:
    """Application constants."""
    
    ROUTING_INFO = {
        "course": "🎓 Course-specific response:",
        "program": "📚 Program-specific response:",
        "both": "🏫 General education response:"
    }
    
    EXAMPLE_QUERIES = [
        "What is the Applied Data Science program about?",
        "What are the prerequisites for Applied Machine Learning?",
        "Tell me about courses in the Master's Program in Management.",
        "List all master's programs in the School of Business, Economics and Law.",
        "What programs are available in Computer Science?"
    ]
    
    SUPPORTED_FILE_ENCODINGS = ['utf-8', 'iso-8859-1', 'latin1']
    SUPPORTED_FILE_EXTENSIONS = {
        'markdown': ['.md'],
        'pdf': ['.pdf']
    }
    
    BATCH_SIZE = 50  # For processing documents in batches

def setup_telemetry():
    """Set up telemetry environment variables to prevent warnings."""
    # Set LangChain telemetry environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = str(Config.LANGCHAIN_TRACING_V2).lower()
    os.environ["ANONYMIZED_TELEMETRY"] = str(Config.ANONYMIZED_TELEMETRY).lower()
    os.environ["POSTHOG_DISABLED"] = str(Config.POSTHOG_DISABLED).lower()
    
    # Set ChromaDB telemetry environment variables
    os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"
    os.environ["CHROMA_TELEMETRY"] = "false"
    
    # Additional telemetry controls
    os.environ["DO_NOT_TRACK"] = "1"

def validate_config():
    """Validate that required configuration is present."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found in environment variables")
    
    # Setup telemetry to prevent warnings
    setup_telemetry()
    
    return True 