import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads
from operator import itemgetter
import gradio as gr
import time
from datetime import datetime
import json
import numpy as np

# Load environment variables
load_dotenv()

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
SYSTEM_TEMPLATE = """You are a helpful course information assistant for Gothenburg University's Computer Science and Engineering department. Your role is to provide accurate information about courses based ONLY on the provided course documents.

Important rules to follow:
1. Only answer questions about courses that are explicitly mentioned in the provided documents
2. If a course is not in the documents, clearly state that you don't have information about that course
3. Base your answers solely on the content from the course documents
4. If you're unsure about any information, say so explicitly
5. When discussing course content, prerequisites, or evaluation methods, quote directly from the source documents when possible
6. Include the course code (e.g., DIT123) when referring to courses

Context from documents: {context}

Current conversation history: {chat_history}

Human question: {question}

Please provide a response based strictly on the above context. If the information isn't in the context, say so."""

class RAGModel:
    def __init__(self, pdf_paths: List[str], md_paths: List[str]):
        """Initialize the RAG model with paths to PDF and Markdown documents."""
        self.pdf_paths = pdf_paths
        self.md_paths = md_paths
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
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = None
        
        # Initialize conversation memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=3
        )
        
        # Create prompt template
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        self.prompt = ChatPromptTemplate.from_messages(messages)
        
        # Initialize conversation chain
        self.qa_chain = None

        # Add multi-query generation template
        self.query_generation_template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant course information from a vector 
        database. Focus on academic and educational terms that might be used in course documentation.
        By generating multiple perspectives on the user question, your goal is to help overcome limitations 
        of distance-based similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
        
        self.query_generation_prompt = ChatPromptTemplate.from_template(self.query_generation_template)
        
        # Setup query generation chain
        self.generate_queries = (
            self.query_generation_prompt 
            | ChatOpenAI(temperature=0, openai_api_key=self.openai_api_key) 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

    def get_unique_union(self, documents: list[list]):
        """Unique union of retrieved docs"""
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    def load_documents(self):
        """Load and process PDF and Markdown documents."""
        try:
            start_time = time.time()
            
            # Check if Chroma database already exists
            if os.path.exists("./data/chroma") and os.listdir("./data/chroma"):
                print("Loading existing embeddings from Chroma database...")
                self.vector_store = Chroma(
                    persist_directory="./data/chroma",
                    embedding_function=self.embeddings
                )
                collection_size = len(self.vector_store.get()['ids'])
                print(f"Loaded {collection_size} existing document chunks from database.")
            else:
                print("Creating new embeddings (this will incur OpenAI API costs)...")

                # Load PDF documents
                documents = []
                for pdf_path in self.pdf_paths:
                    pdf_loader = DirectoryLoader(
                        pdf_path,
                        glob="**/*.pdf",
                        loader_cls=PyPDFLoader,
                        show_progress=True
                    )
                    documents.extend(pdf_loader.load())

                # Load Markdown documents
                for md_path in self.md_paths:
                    md_loader = DirectoryLoader(
                        md_path,
                        glob="**/*.md",
                        loader_cls=TextLoader,
                        loader_kwargs={'autodetect_encoding': True},
                        show_progress=True
                    )
                    try:
                        md_docs = md_loader.load()
                        documents.extend(md_docs)
                    except Exception as e:
                        print(f"Error loading markdown from {md_path}: {str(e)}")
                        continue

                chunks = self.text_splitter.split_documents(documents)
                
                # Create new Chroma instance with persistence
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory="./data/chroma",
                    collection_metadata={"hnsw:space": "cosine"}
                )
                
                collection_size = len(chunks)
                print("Embeddings created and stored in Chroma database.")
            
            # Initialize QA chain with specific configuration
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 10,
                        "fetch_k": 15
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

    def query(self, question: str) -> Dict:
        """Process a user query using multi-query retrieval while maintaining context and chat history."""
        if not self.qa_chain:
            raise ValueError("Model not initialized. Please load documents first.")
        
        start_time = time.time()
        try:
            # Setup retriever
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 15}
            )
            
            # Generate multiple queries and get unique documents
            retrieval_chain = self.generate_queries | retriever.map() | self.get_unique_union
            docs = retrieval_chain.invoke({"question": question})
            
            # Prepare the context from retrieved documents
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # Get chat history
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            # Create the combined prompt with context and chat history
            messages = [
                SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
            prompt = ChatPromptTemplate.from_messages(messages)
            
            # Generate response using the LLM
            response = self.llm.invoke(
                prompt.format(
                    context=context,
                    chat_history=chat_history,
                    question=question
                )
            )
            
            # Update memory
            self.memory.save_context({"question": question}, {"answer": response.content})
            
            # Format source information
            sources = []
            if docs:
                for doc in docs:
                    source_name = os.path.basename(doc.metadata.get("source", ""))
                    if source_name and source_name not in sources:
                        sources.append(source_name)
            
            # Add source information to the answer
            answer = response.content
            if sources:
                answer += f"\n\nSources: {', '.join(sources)}"
            
            # Update metrics
            query_time = time.time() - start_time
            self.metrics.update_query_metrics(True, query_time)
            self.metrics.update_retrieval_metrics(docs)
            
            return {
                "answer": answer,
                "source_documents": docs
            }
            
        except Exception as e:
            query_time = time.time() - start_time
            self.metrics.update_query_metrics(False, query_time)
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": []
            }

def create_gradio_interface(rag_model: RAGModel):
    """Create and launch the Gradio interface."""
    
    def process_query(message: str, history: List[List[str]]) -> str:
        try:
            response = rag_model.query(message)
            return response["answer"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_metrics():
        return rag_model.metrics.get_metrics_summary()

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # GuPT: Course and Programme Information Assistant
        Ask questions about Gothenburg University's courses and programmes.
        """)
        
        with gr.Row(equal_height=True):
            # Chat column (2/3 of width)
            with gr.Column(scale=2):
                chat_interface = gr.ChatInterface(
                    fn=process_query,
                    examples=[
                        "What is the Deep Machine Learning course about?",
                        "What are the prerequisites for DIT227?",
                        "Tell me about the evaluation methods in DIT235."
                    ],
                    css="div.message-wrap { height: 600px !important; overflow-y: auto; }",
                    type="messages"
                )
            
            # Metrics column (1/3 of width)
            with gr.Column(scale=1):
                # System Metrics Section
                gr.Markdown("### System Metrics")
                metrics_display = gr.Markdown()
                refresh_btn = gr.Button("Refresh Metrics", variant="primary")
                refresh_btn.click(get_metrics, outputs=metrics_display)
                
                # Initialize metrics display
                metrics_display.value = get_metrics()
                
                # System Information Section
                gr.Markdown("### System Information")
                gr.Markdown("""
                - **Model**: GPT-4o-mini
                - **Embedding Model**: OpenAI Embeddings
                - **Vector Store**: Chroma
                - **Search Strategy**: MMR (Maximum Marginal Relevance)
                - **Temperature**: 0.1
                """)
    
    return interface

def main():
    # Initialize RAG model
    pdf_paths = ["data/courses/pdf/", "data/programs/pdf/"]
    md_paths = ["data/courses/md/", "data/programs/md/"]
    
    rag_model = RAGModel(pdf_paths, md_paths)
    
    # Load and process documents
    print("Loading and processing documents...")
    num_chunks = rag_model.load_documents()
    print(f"Processed {num_chunks} document chunks")
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(rag_model)
    interface.launch(share=False)

if __name__ == "__main__":
    main()
