import os
import time
from typing import List, Dict, Any, Optional

# LangChain imports using modern patterns
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage

# Local imports
from config import Config, PromptTemplates, validate_config
from models import RouteQuery, QueryResult, RetrievalStats
from document_processor import DocumentProcessor
from chat_logger import ChatLogger

class RAGService:
    """Modern RAG service using LangChain Expression Language (LCEL)."""
    
    def __init__(self, base_path: str = None):
        """Initialize the RAG service.
        
        Args:
            base_path: Base path for documents and vector store
        """
        # Validate configuration
        validate_config()
        
        self.base_path = base_path or Config.DATA_BASE_PATH
        self.chat_logger = ChatLogger()
        self.conversation_memory = []  # Simple in-memory conversation storage
        
        # Initialize components
        self._initialize_models()
        self._initialize_vector_store()
        self._setup_chains()
        
        # Track last generated queries for logging
        self.last_generated_queries = []
        
    def _initialize_models(self):
        """Initialize LLM and embedding models."""
        print("Initializing AI models...")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize embeddings with better error handling
        self.embeddings = OpenAIEmbeddings(
            api_key=Config.OPENAI_API_KEY,
            model=Config.EMBEDDING_MODEL,
            chunk_size=Config.EMBEDDING_CHUNK_SIZE,
            max_retries=Config.EMBEDDING_MAX_RETRIES,
            request_timeout=Config.EMBEDDING_REQUEST_TIMEOUT
        )
        
        print("✅ AI models initialized successfully")
    
    def _initialize_vector_store(self):
        """Initialize the vector store (empty initially)."""
        self.vector_store = None
        print("Vector store placeholder initialized")
    
    def _setup_chains(self):
        """Set up all the LCEL chains."""
        print("Setting up LangChain LCEL chains...")
        
        # Router chain
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", PromptTemplates.ROUTER_SYSTEM_TEMPLATE),
            ("human", "{question}")
        ])
        self.router_chain = router_prompt | self.llm.with_structured_output(RouteQuery)
        
        # Query generation chains for different content types
        self.query_generation_chains = {}
        for content_type in ["course", "program", "both"]:
            template = PromptTemplates.get_query_template(content_type)
            prompt = ChatPromptTemplate.from_template(template)
            self.query_generation_chains[content_type] = prompt | self.llm | StrOutputParser()
        
        # Main QA chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", PromptTemplates.SYSTEM_TEMPLATE),
            ("human", "{question}")
        ])
        
        # This will be completed when vector store is loaded
        self.qa_chain = None
        
        print("✅ LCEL chains set up successfully")
    
    def load_documents(self) -> int:
        """Load and process documents, create or load vector store.
        
        Returns:
            Number of document chunks processed
        """
        try:
            print(f"Checking for existing database at: {Config.CHROMA_DB_PATH}")
            
            if os.path.exists(Config.CHROMA_DB_PATH) and os.listdir(Config.CHROMA_DB_PATH):
                print("Existing database found, attempting to load...")
                count = self._load_existing_database()
                if count == 0:
                    print("⚠️  Existing database is empty, rebuilding...")
                    return self._create_new_database()
                return count
            else:
                print("No existing database found, creating new one...")
                return self._create_new_database()
                
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise
    
    def _load_existing_database(self) -> int:
        """Load existing vector database.
        
        Returns:
            Number of documents in the database
        """
        print("Loading existing embeddings from Chroma database...")
        
        try:
            self.vector_store = Chroma(
                persist_directory=Config.CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"},
                collection_name=Config.COLLECTION_NAME
            )
            
            # Get collection size
            collection_data = self.vector_store.get()
            collection_size = len(collection_data['ids'])
            
            if collection_size == 0:
                print("Database exists but is empty")
                return 0
            
            print(f"✅ Loaded {collection_size} existing document chunks from database")
            self._setup_qa_chain()
            return collection_size
            
        except Exception as e:
            print(f"Error loading existing database: {str(e)}")
            return 0
    
    def _create_new_database(self) -> int:
        """Create new vector database from documents.
        
        Returns:
            Number of document chunks processed
        """
        print("Creating new embeddings (this will incur OpenAI API costs)...")
        
        # Process documents
        processor = DocumentProcessor(self.base_path)
        documents = processor.process_all_documents()
        
        if not documents:
            raise ValueError("No documents found to process")
        
        # Chunk documents
        chunks = processor.chunk_documents(documents)
        
        # Initialize empty vector store
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_DB_PATH,
            collection_metadata={"hnsw:space": "cosine"},
            collection_name=Config.COLLECTION_NAME
        )
        
        # Process documents in batches to avoid token limits
        total_processed = self._process_documents_in_batches(chunks)
        
        print(f"✅ Database creation completed! Processed {total_processed} documents.")
        self._setup_qa_chain()
        return total_processed
    
    def _process_documents_in_batches(self, chunks: List[Document]) -> int:
        """Process documents in batches to avoid API limits.
        
        Args:
            chunks: List of document chunks to process
            
        Returns:
            Number of successfully processed chunks
        """
        batch_size = Config.BATCH_SIZE
        total_processed = 0
        
        print(f"Processing {len(chunks)} document chunks in batches of {batch_size}...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                self.vector_store.add_documents(batch)
                total_processed += len(batch)
                print(f"✅ Successfully processed {len(batch)} documents (Total: {total_processed})")
                
                # Small delay to be nice to the API
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_num}: {str(e)}")
                # Continue with next batch instead of failing completely
                continue
        
        return total_processed
    
    def _setup_qa_chain(self):
        """Set up the main QA chain with retriever."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # Create the main QA chain using LCEL
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", PromptTemplates.SYSTEM_TEMPLATE),
            ("human", "{question}")
        ])
        
        def format_docs(docs):
            """Format retrieved documents for the prompt."""
            return "\n\n".join([d.page_content for d in docs])
        
        def format_chat_history(memory):
            """Format chat history for the prompt."""
            if not memory:
                return "No previous conversation."
            
            formatted = []
            for msg in memory[-6:]:  # Last 6 messages (3 exchanges)
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = msg.type
                    content = msg.content
                else:
                    continue
                formatted.append(f"{role}: {content}")
            
            return "\n".join(formatted)
        
        # Create retriever (will be configured per query)
        self.base_retriever = self.vector_store.as_retriever()
        
        # The QA chain will be constructed per query with specific retriever config
        self.qa_prompt = qa_prompt
        self.format_docs = format_docs
        self.format_chat_history = format_chat_history
        
        print("✅ QA chain set up successfully")
    
    def route_query(self, question: str) -> str:
        """Route the query to determine content type.
        
        Args:
            question: User's question
            
        Returns:
            Content type: 'course', 'program', or 'both'
        """
        try:
            result = self.router_chain.invoke({"question": question})
            return result.content_type
        except Exception as e:
            print(f"Error in query routing: {str(e)}")
            return "both"  # Default to both if routing fails
    
    def generate_query_variations(self, question: str, content_type: str) -> List[str]:
        """Generate multiple query variations for better retrieval.
        
        Args:
            question: Original question
            content_type: Content type from routing
            
        Returns:
            List of query variations
        """
        try:
            chain = self.query_generation_chains[content_type]
            variations = chain.invoke({"question": question})
            
            # Process and clean the variations
            queries = [q.strip() for q in variations.split('\n') if q.strip()]
            
            # Always include the original question
            if question not in queries:
                queries.append(question)
            
            # Store for logging
            self.last_generated_queries = queries
            
            return queries
            
        except Exception as e:
            print(f"Error generating query variations: {str(e)}")
            # Fallback to original question
            self.last_generated_queries = [question]
            return [question]
    
    def retrieve_documents(self, question: str, content_type: str) -> List[Document]:
        """Retrieve relevant documents using multiple query variations.
        
        Args:
            question: Original question
            content_type: Content type from routing
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        # Generate query variations
        queries = self.generate_query_variations(question, content_type)
        
        print(f"\nGenerated queries for '{question}':")
        for q in queries:
            print(f"  • {q}")
        
        # Configure retriever based on content type
        k = Config.RETRIEVAL_K_VALUES[content_type]
        
        # Create metadata filter if needed
        search_kwargs = {
            "k": k,
            "fetch_k": k * 3  # Fetch more candidates for MMR
        }
        
        if content_type != "both":
            search_kwargs["filter"] = {"doc_type": content_type}
        
        # Configure retriever
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        
        # Retrieve documents for each query variation
        all_docs = []
        for query in queries:
            try:
                docs = retriever.invoke(query)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error retrieving for query '{query}': {str(e)}")
                continue
        
        # Remove duplicates while preserving order
        unique_docs = []
        seen_content = set()
        
        for doc in all_docs:
            # Create a unique identifier from content and source
            doc_id = f"{doc.page_content[:100]}_{doc.metadata.get('source', '')}"
            if doc_id not in seen_content:
                seen_content.add(doc_id)
                unique_docs.append(doc)
        
        # Log retrieval statistics
        doc_types = [doc.metadata.get('doc_type', 'unknown') for doc in unique_docs]
        print(f"\nRetrieved {len(unique_docs)} unique documents:")
        print(f"  • Courses: {doc_types.count('course')}")
        print(f"  • Programs: {doc_types.count('program')}")
        
        return unique_docs
    
    def query(self, question: str) -> QueryResult:
        """Process a user query and return response.
        
        Args:
            question: User's question
            
        Returns:
            QueryResult with answer and metadata
        """
        if not self.vector_store:
            raise ValueError("Model not initialized. Please load documents first.")
        
        start_time = time.time()
        
        try:
            # Route the query
            content_type = self.route_query(question)
            print(f"Query routed as: {content_type}")
            
            # Retrieve relevant documents
            docs = self.retrieve_documents(question, content_type)
            
            # Format context and chat history
            context = self.format_docs(docs)
            chat_history = self.format_chat_history(self.conversation_memory)
            
            # Generate answer using LCEL
            chain = self.qa_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "question": question,
                "chat_history": chat_history
            })
            
            # Update conversation memory
            self.conversation_memory.extend([
                {"role": "human", "content": question},
                {"role": "assistant", "content": answer}
            ])
            
            # Keep memory within reasonable size
            if len(self.conversation_memory) > 12:  # Keep last 6 exchanges
                self.conversation_memory = self.conversation_memory[-12:]
            
            # Format sources
            sources = self._format_sources(docs)
            if sources:
                answer += sources
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = QueryResult(
                answer=answer,
                source_documents=docs,
                content_type=content_type,
                processing_time=processing_time,
                generated_queries=self.last_generated_queries
            )
            
            # Log the interaction
            self._log_interaction(question, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return QueryResult(
                answer=error_msg,
                source_documents=[],
                content_type="error",
                processing_time=time.time() - start_time
            )
    
    def _format_sources(self, docs: List[Document]) -> str:
        """Format source documents for display.
        
        Args:
            docs: Retrieved documents
            
        Returns:
            Formatted sources string
        """
        if not docs:
            return ""
        
        # Get unique sources
        sources = list(set(
            os.path.basename(doc.metadata.get("source", ""))
            for doc in docs if doc.metadata.get("source")
        ))
        sources = sorted(sources)
        
        if not sources:
            return ""
        
        sources_text = ""
        if len(sources) > 2:
            # Show only first 2 sources with expandable section for more
            visible_sources = sources[:2]
            hidden_sources = sources[2:]
            sources_text += "\n\nSources:"
            for source in visible_sources:
                sources_text += f"\n• {source}"
            sources_text += f"\n<details><summary>**See {len(hidden_sources)} more sources...**</summary>\n"
            for source in hidden_sources:
                sources_text += f"\n• {source}"
            sources_text += "\n</details>"
        else:
            # If 2 or fewer sources, show all
            sources_text += "\n\nSources:"
            for source in sources:
                sources_text += f"\n• {source}"
        
        return sources_text
    
    def _log_interaction(self, question: str, result: QueryResult):
        """Log the interaction for analysis.
        
        Args:
            question: User's question
            result: Query result
        """
        try:
            system_info = {
                "model_version": Config.MODEL_NAME,
                "embedding_version": Config.EMBEDDING_MODEL,
                "search_config": {
                    "search_type": "mmr",
                    "k_value": Config.RETRIEVAL_K_VALUES.get(result.content_type),
                    "content_type": result.content_type
                }
            }
            
            self.chat_logger.log_interaction(
                question=question,
                answer=result.answer,
                source_documents=result.source_documents,
                content_type=result.content_type,
                generated_queries=result.generated_queries or [],
                processing_time=result.processing_time or 0,
                chat_history=self.conversation_memory,
                system_info=system_info
            )
        except Exception as e:
            print(f"Error logging interaction: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status.
        
        Returns:
            Dictionary with system status information
        """
        status = {
            "database_initialized": self.vector_store is not None,
            "model_version": Config.MODEL_NAME,
            "embedding_version": Config.EMBEDDING_MODEL,
            "conversation_length": len(self.conversation_memory),
            "last_queries": self.last_generated_queries
        }
        
        if self.vector_store:
            try:
                collection_data = self.vector_store.get()
                status["documents_loaded"] = len(collection_data['ids'])
            except:
                status["documents_loaded"] = "unknown"
        else:
            status["documents_loaded"] = 0
        
        return status
    
    def clear_conversation_memory(self):
        """Clear the conversation memory."""
        self.conversation_memory = []
        print("Conversation memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_memory.copy() 