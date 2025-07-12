import gradio as gr
from typing import List, Dict, Any
from rag_service import RAGService
from config import Config, AppConstants

class RAGInterface:
    """Gradio interface for the RAG application."""
    
    def __init__(self, rag_service: RAGService):
        """Initialize the interface.
        
        Args:
            rag_service: The RAG service instance
        """
        self.rag_service = rag_service
        self.interface = None
    
    def process_query(self, message: str, history: List[Dict[str, str]]) -> str:
        """Process a single query in the chat interface.
        
        Args:
            message: User's message
            history: Chat history in OpenAI-style format with 'role' and 'content' keys
            
        Returns:
            Assistant's response
        """
        try:
            # Query the RAG service
            result = self.rag_service.query(message)
            
            # Format response with routing information
            content_type = result.content_type
            answer = result.answer
            
            # Add routing indicator
            routing_prefix = AppConstants.ROUTING_INFO.get(content_type, "")
            if routing_prefix:
                return f"{routing_prefix}\n\n{answer}"
            else:
                return answer
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"Interface error: {error_msg}")
            return error_msg
    
    def get_system_info(self) -> str:
        """Get formatted system information.
        
        Returns:
            Formatted system status string
        """
        try:
            status = self.rag_service.get_system_status()
            
            # Format the information nicely
            info = f"""
### 📊 System Status

**Database Status:** {'✅ Initialized' if status['database_initialized'] else '❌ Not Initialized'}
**Documents Loaded:** {status['documents_loaded']}
**Model Version:** {status['model_version']}
**Embedding Model:** {status['embedding_version']}
**Conversation Length:** {status['conversation_length']} messages

### 🔍 Search Configuration

**Course Queries:** Top {Config.RETRIEVAL_K_VALUES['course']} matches
**Program Queries:** Top {Config.RETRIEVAL_K_VALUES['program']} matches  
**Mixed Queries:** Top {Config.RETRIEVAL_K_VALUES['both']} matches
**Search Type:** MMR (Maximal Marginal Relevance)

### 📚 Query Types

**🎓 Course Queries**
- Specific course information
- Prerequisites and requirements
- Learning outcomes
- Course content and structure

**📚 Program Queries**
- Program overviews and structure
- Available programs by department
- Program requirements and outcomes
- Career opportunities

**🏫 General Queries**
- Courses within programs
- Department offerings
- Combined course/program information
- Cross-referencing content
"""
            return info.strip()
            
        except Exception as e:
            return f"Error getting system info: {str(e)}"
    

    

    
    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        # Create the interface
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # GuPT: Gothenburg University Information Assistant
            Ask questions about Gothenburg University's courses and programs.
            """)
            
            with gr.Row(equal_height=True):
                # Chat column (2/3 of width)
                with gr.Column(scale=2):
                    chat_interface = gr.ChatInterface(
                        fn=self.process_query,
                        examples=AppConstants.EXAMPLE_QUERIES,
                        css="""
                        div.message-wrap { height: 600px !important; overflow-y: auto; }
                        details { margin-top: 10px; }
                        summary { cursor: pointer; color: #2A6BB0; }
                        summary:hover { text-decoration: underline; }
                        """,
                        type="messages"
                    )
                
                # Info column (1/3 of width)
                with gr.Column(scale=1):
                    # Get system status for static display
                    status = self.rag_service.get_system_status()
                    
                    gr.Markdown(f"""
                    ### Document Collection
                    - Documents Loaded: {status['documents_loaded']}
                    - Database Status: {'✅ Initialized' if status['database_initialized'] else '❌ Not Ready'}
                    - Model: {status['model_version']}
                    
                    ### Search Configuration
                    - Using MMR for diverse results
                    - Course queries: top {Config.RETRIEVAL_K_VALUES['course']} matches
                    - Program queries: top {Config.RETRIEVAL_K_VALUES['program']} matches
                    - Mixed queries: top {Config.RETRIEVAL_K_VALUES['both']} matches
                    
                    ### Query Types
                    
                    🎓 **Course Queries**
                    - Specific course information
                    - Prerequisites and requirements
                    - Learning outcomes
                    
                    📚 **Program Queries**
                    - Program overviews
                    - Available programs by department
                    - Program requirements
                    
                    🏫 **General Queries**
                    - Courses within programs
                    - Department offerings
                    - Combined course/program information
                    """)
        
        self.interface = interface
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface.
        
        Args:
            **kwargs: Additional arguments for Gradio launch
        """
        if not self.interface:
            self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "share": False,
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "show_error": True,
            "quiet": False
        }
        
        # Update with any provided parameters
        launch_params.update(kwargs)
        
        print(f"🚀 Launching GuPT interface...")
        print(f"📍 Server: {launch_params['server_name']}:{launch_params['server_port']}")
        
        try:
            self.interface.launch(**launch_params)
        except Exception as e:
            print(f"❌ Error launching interface: {str(e)}")
            raise

def create_interface(rag_service: RAGService) -> RAGInterface:
    """Factory function to create a RAG interface.
    
    Args:
        rag_service: The RAG service instance
        
    Returns:
        Configured RAGInterface instance
    """
    return RAGInterface(rag_service) 