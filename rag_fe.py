import os
from typing import List, Dict, Literal
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
import gradio as gr
import time
from datetime import datetime
import json
import numpy as np
from evaluation import EvaluationSystem, ResponseMetrics

# Load environment variables
load_dotenv()

# Define the routing model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant content type."""
    content_type: Literal["course", "program", "both"] = Field(
        ...,
        description="Route to: 'course' for specific course questions, 'program' for program questions, 'both' when the question involves both or is unclear"
    )

# Define specialized prompt templates for different content types
COURSE_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents about university COURSES. 
Focus on aspects like course content, prerequisites, learning outcomes, examination methods, and specific details.
Provide these alternative questions separated by newlines.

Original question: {question}"""

PROGRAM_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents about university PROGRAMS. 
Focus on aspects like program structure, career opportunities, admission requirements, and overall outcomes.
Provide these alternative questions separated by newlines.

Original question: {question}"""

GENERAL_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents about both university COURSES and PROGRAMS. 
Ensure a balanced focus on both course-specific details and program-level information.
Provide these alternative questions separated by newlines.

Original question: {question}"""

# Define the routing prompt
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

def get_unique_union(documents: List[List[Document]]) -> List[Document]:
    """Get unique union of retrieved documents."""
    seen = set()
    unique_docs = []
    
    for doc_list in documents:
        for doc in doc_list:
            # Create a unique identifier from content and metadata
            doc_id = f"{doc.page_content}_{json.dumps(doc.metadata, sort_keys=True)}"
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
    
    return unique_docs

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0,
            'response_times': [],
            'retrieval_stats': {
                'total_retrievals': 0,
                'avg_chunks_retrieved': 0,
                'unique_docs_accessed': set()
            },
            'session_start': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_distribution': {}
        }
    
    def update_query_metrics(self, success: bool, response_time: float):
        self.metrics['total_queries'] += 1
        if success:
            self.metrics['successful_queries'] += 1
        else:
            self.metrics['failed_queries'] += 1
        
        self.metrics['response_times'].append(response_time)
        self.metrics['avg_response_time'] = np.mean(self.metrics['response_times'])
    
    def update_retrieval_metrics(self, source_docs: List):
        self.metrics['retrieval_stats']['total_retrievals'] += 1
        self.metrics['retrieval_stats']['avg_chunks_retrieved'] = (
            (self.metrics['retrieval_stats']['avg_chunks_retrieved'] * 
             (self.metrics['retrieval_stats']['total_retrievals'] - 1) + 
             len(source_docs)) / self.metrics['retrieval_stats']['total_retrievals']
        )
        
        for doc in source_docs:
            source = os.path.basename(doc.metadata.get('source', ''))
            if source:
                self.metrics['retrieval_stats']['unique_docs_accessed'].add(source)
                self.metrics['source_distribution'][source] = self.metrics['source_distribution'].get(source, 0) + 1
    
    def get_metrics_summary(self) -> str:
        success_rate = (self.metrics['successful_queries'] / self.metrics['total_queries'] * 100 
                       if self.metrics['total_queries'] > 0 else 0)
        
        summary = f"""
### RAG System Performance Metrics

#### Query Statistics
- Total Queries: {self.metrics['total_queries']}
- Success Rate: {success_rate:.1f}%
- Average Response Time: {self.metrics['avg_response_time']:.2f}s

#### Retrieval Statistics
- Total Documents Retrieved: {self.metrics['retrieval_stats']['total_retrievals']}
- Average Chunks per Query: {self.metrics['retrieval_stats']['avg_chunks_retrieved']:.1f}
- Unique Documents Accessed: {len(self.metrics['retrieval_stats']['unique_docs_accessed'])}

#### Session Information
- Start Time: {self.metrics['session_start']}
- Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return summary

# Define the system prompt template
SYSTEM_TEMPLATE = """You are a helpful course and program information assistant for Gothenburg University. 
Your role is to provide accurate information about courses and programs based ONLY on the provided course and program documents.

Important rules to follow:
1. Only answer questions about courses that are explicitly mentioned in the provided documents
2. If a course is not in the documents, clearly state that you don't have information about that course
3. Base your answers solely on the content from the course documents
4. If you're unsure about any information, say so explicitly
5. When discussing course content, prerequisites, or evaluation methods, quote directly from the source documents when possible
6. Include the course code (e.g., DIT123) when referring to courses
7. When listing programs or courses, ensure to enumerate ALL items found in the provided documents, not just a subset
8. For questions asking about all programs from a specific school/department, make sure to list every program mentioned in the context

Context from documents: {context}

Current conversation history: {chat_history}

Human question: {question}

Please provide a response based strictly on the above context. If the information isn't in the context, say so."""

class RAGModel:
    def __init__(self, pdf_folder_path: str):
        """Initialize the RAG model with the path to PDF documents."""
        self.pdf_folder_path = pdf_folder_path
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker()
        
        # Initialize OpenAI model
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model="text-embedding-3-small")
        
        # Initialize text splitter with optimized settings for course PDFs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = None
        
        # Initialize conversation memory with window size
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=3  # Remember last 3 interactions
        )
        
        # Initialize routing components
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_TEMPLATE),
            ("human", "{question}")
        ])
        self.router = self.router_prompt | self.llm.with_structured_output(RouteQuery)
        
        # Initialize specialized query prompts
        self.query_prompts = {
            "course": ChatPromptTemplate.from_template(COURSE_QUERY_TEMPLATE),
            "program": ChatPromptTemplate.from_template(PROGRAM_QUERY_TEMPLATE),
            "both": ChatPromptTemplate.from_template(GENERAL_QUERY_TEMPLATE)
        }
        
        # Create answer prompt template
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        self.prompt = ChatPromptTemplate.from_messages(messages)
        
        # Initialize conversation chain
        self.qa_chain = None
        
        # Initialize evaluation system
        self.evaluation_system = EvaluationSystem()
        
        # Add model version tracking
        self.model_version = "gpt-4-0125-preview"
        self.embedding_version = "text-embedding-3-small"

    def route_query(self, question: str) -> str:
        """Route the query to the appropriate content type."""
        result = self.router.invoke({"question": question})
        return result.content_type

    def generate_queries(self, question: str, content_type: str) -> List[str]:
        """Generate multiple versions of the input question based on content type."""
        prompt = self.query_prompts[content_type]
        chain = prompt | self.llm | StrOutputParser()
        queries = chain.invoke({"question": question})
        return [q.strip() for q in queries.split('\n') if q.strip()]

    def retrieve_documents(self, question: str, content_type: str) -> List[Document]:
        """Retrieve documents using multiple queries and content type."""
        # Generate multiple queries based on content type
        queries = self.generate_queries(question, content_type)
        
        # Adjust retrieval parameters based on content type
        k_values = {
            "course": 6,     
            "program": 15,  
            "both": 15        
        }
        
        # Get documents for each query
        all_docs = []
        for query in queries:
            docs = self.vector_store.similarity_search(
                query,
                k=k_values[content_type],
            )
            all_docs.append(docs)
        
        # Get unique union of documents
        unique_docs = get_unique_union(all_docs)
        
        # Update metrics
        self.metrics.update_retrieval_metrics(unique_docs)
        
        return unique_docs

    def query(self, question: str) -> Dict:
        """Process a user query and return the response."""
        if not self.qa_chain:
            raise ValueError("Model not initialized. Please load documents first.")
        
        start_time = time.time()
        try:
            # First, route the query
            content_type = self.route_query(question)
            
            # Retrieve documents using multi-query approach with content type
            docs = self.retrieve_documents(question, content_type)
            
            # Format documents for the prompt
            docs_content = "\n\n".join([d.page_content for d in docs])
            
            # Generate answer using the retrieved documents
            chain = (
                self.prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            answer = chain.invoke({
                "context": docs_content,
                "question": question,
                "chat_history": self.memory.chat_memory.messages
            })
            
            # Evaluate and validate the response
            evaluation_metrics = self.evaluation_system.evaluate_response(
                query=question,
                response=answer,
                contexts=[doc.page_content for doc in docs],
                response_type=content_type,
                model_version=self.model_version,
                embedding_version=self.embedding_version
            )
            
            # Update memory
            self.memory.save_context({"question": question}, {"answer": answer})
            
            # Format source information
            sources = []
            for doc in docs:
                source_name = os.path.basename(doc.metadata.get("source", ""))
                if source_name and source_name not in sources:
                    sources.append(source_name)
            
            # Add source and routing information to the answer
            if sources:
                answer += f"\n\nSources: {', '.join(sources)}"
            
            # Update metrics
            query_time = time.time() - start_time
            self.metrics.update_query_metrics(True, query_time)
            
            return {
                "answer": answer,
                "source_documents": docs,
                "content_type": content_type,
                "evaluation_metrics": evaluation_metrics
            }
        except Exception as e:
            query_time = time.time() - start_time
            self.metrics.update_query_metrics(False, query_time)
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "content_type": None,
                "evaluation_metrics": None
            }

    def load_documents(self):
        """Load and process PDF documents."""
        try:
            start_time = time.time()
            
            # Check if Chroma database already exists
            if os.path.exists("./data/chroma") and os.listdir("./data/chroma"):
                print("Loading existing embeddings from Chroma database...")
                self.vector_store = Chroma(
                    persist_directory="./data/chroma",
                    embedding_function=self.embeddings,
                    collection_metadata={"hnsw:space": "cosine"},
                    collection_name="course_docs"
                )
                collection_size = len(self.vector_store.get()['ids'])
                print(f"Loaded {collection_size} existing document chunks from database.")
            else:
                print("Creating new embeddings (this will incur OpenAI API costs)...")
                # Use the new process_documents function instead of DirectoryLoader
                documents = process_documents(".")  # Pass the base path
                chunks = self.text_splitter.split_documents(documents)
                
                # Create new Chroma instance with persistence
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory="./data/chroma",
                    collection_metadata={"hnsw:space": "cosine"},
                    collection_name="course_docs"
                )
                
                collection_size = len(chunks)
                print("Embeddings created and stored in Chroma database.")
            
            # Initialize QA chain with specific configuration
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 8,
                    }
                ),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": self.prompt},
                verbose=True
            )
            
            load_time = time.time() - start_time
            print(f"Database loaded in {load_time:.2f} seconds")
            
            return collection_size
            
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise

def create_gradio_interface(rag_model: RAGModel):
    """Create and launch the Gradio interface."""
    
    def process_query(message: str, history: List[List[str]]) -> str:
        try:
            response = rag_model.query(message)
            content_type = response["content_type"]
            answer = response["answer"]
            
            # Add evaluation metrics to the response
            if response.get("evaluation_metrics"):
                metrics = response["evaluation_metrics"]
                eval_info = f"\n\n---\nResponse Quality Metrics:"
                if metrics.semantic_similarity is not None:
                    eval_info += f"\n- Semantic Similarity: {metrics.semantic_similarity:.3f}"
                if metrics.context_relevance is not None:
                    eval_info += f"\n- Context Relevance: {metrics.context_relevance:.3f}"
                answer += eval_info
            
            # Add routing information
            routing_info = {
                "course": "🎓 Course-specific response:",
                "program": "📚 Program-specific response:",
                "both": "🏫 General education response:"
            }
            
            return f"{routing_info.get(content_type, '')} \n\n{answer}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_metrics():
        """Get both system metrics and evaluation metrics"""
        system_metrics = rag_model.metrics.get_metrics_summary()
        evaluation_metrics = rag_model.evaluation_system.get_evaluation_summary()
        return f"{system_metrics}\n\n{evaluation_metrics}"

    def run_evaluation():
        """Run the full evaluation suite"""
        print("Running evaluation suite...")
        metrics = rag_model.evaluation_system.run_test_suite(rag_model)
        return rag_model.evaluation_system.get_evaluation_summary()

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # G(U)PT: Gothenburg University Information Assistant
        Ask questions about Gothenburg University's courses and programs.
        """)
        
        with gr.Row(equal_height=True):
            # Chat column (2/3 of width)
            with gr.Column(scale=2):
                chat_interface = gr.ChatInterface(
                    fn=process_query,
                    examples=[
                        "What is the Applied Data Science program about?",
                        "What are the prerequisites for Applied Machine Learning?",
                        "Tell me about courses in the Master's Program in Management."
                    ],
                    css="div.message-wrap { height: 600px !important; overflow-y: auto; }",
                    type="messages"
                )
            
            # Metrics column (1/3 of width)
            with gr.Column(scale=1):
                # System Metrics Section
                gr.Markdown("### System Metrics")
                metrics_display = gr.Markdown()
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Metrics", variant="primary")
                    evaluate_btn = gr.Button("Run Full Evaluation", variant="secondary")
                
                refresh_btn.click(get_metrics, outputs=metrics_display)
                evaluate_btn.click(run_evaluation, outputs=metrics_display)
                
                # Initialize metrics display
                metrics_display.value = get_metrics()
                
                # System Information Section
                gr.Markdown("### System Information")
                gr.Markdown("""
                - **Model**: GPT-4-0125-preview
                - **Embedding Model**: text-embedding-3-small
                - **Vector Store**: Chroma
                - **Search Strategy**: MMR with Content Routing
                - **Temperature**: 0.1
                
                ### Content Types
                - 🎓 Course-specific information
                - 📚 Program-specific information
                - 🏫 General education queries
                """)

    return interface

def process_documents(base_path):
    """Process both markdown and PDF documents from courses and programs directories"""
    documents = []
    
    # Process markdown files
    md_paths = {
        'courses': os.path.join(base_path, 'data/courses/md'),
        'programs': os.path.join(base_path, 'data/programs/md')
    }
    
    # Process PDF files 
    pdf_paths = {
        'courses': os.path.join(base_path, 'data/courses/pdf'),
        'programs': os.path.join(base_path, 'data/programs/pdf')
    }

    # Ensure directories exist
    for path_dict in [md_paths, pdf_paths]:
        for path in path_dict.values():
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")

    # Process markdown files
    for category, path in md_paths.items():
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith('.md'):
                    file_path = os.path.join(path, filename)
                    try:
                        # Try UTF-8 first
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        try:
                            # If UTF-8 fails, try with ISO-8859-1
                            with open(file_path, 'r', encoding='iso-8859-1') as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            # If both fail, try with latin1
                            with open(file_path, 'r', encoding='latin1') as f:
                                content = f.read()
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': file_path,
                            'type': 'markdown',
                            'category': category
                        }
                    )
                    documents.append(doc)

    # Process PDF files
    for category, path in pdf_paths.items():
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(path, filename)
                    try:
                        # Use PDF loader
                        loader = PyPDFLoader(file_path)
                        pdf_docs = loader.load()
                        
                        # Add metadata to each page
                        for doc in pdf_docs:
                            doc.metadata.update({
                                'type': 'pdf',
                                'category': category
                            })
                        documents.extend(pdf_docs)
                    except Exception as e:
                        print(f"Error processing PDF {filename}: {str(e)}")
                        continue

    return documents

def main():
    # Initialize RAG model with base path
    rag_model = RAGModel(".")  # Changed from "data/pdf/"
    
    # Load and process documents
    print("Loading and processing documents...")
    num_chunks = rag_model.load_documents()
    print(f"Processed {num_chunks} document chunks")
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(rag_model)
    interface.launch(share=False)

if __name__ == "__main__":
    main()
