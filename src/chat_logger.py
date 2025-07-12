import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from models import ChatInteraction, RetrievalStats
from config import Config

class ChatLogger:
    """Handles logging of chat interactions with enhanced metadata."""
    
    def __init__(self, log_file: str = None):
        """Initialize the chat logger.
        
        Args:
            log_file: Path to the log file. If None, uses config default.
        """
        self.log_file = log_file or Config.LOG_FILE
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Create log file if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def log_interaction(self, 
                       question: str,
                       answer: str,
                       source_documents: List[Any],
                       content_type: str,
                       generated_queries: List[str],
                       processing_time: float,
                       chat_history: List[Any],
                       system_info: Dict[str, Any]) -> None:
        """Log a complete chat interaction with detailed metadata.
        
        Args:
            question: The user's question
            answer: The generated answer
            source_documents: Retrieved documents
            content_type: The routing type (course/program/both)
            generated_queries: List of generated query variations
            processing_time: Time taken to process the query
            chat_history: Chat memory messages
            system_info: System configuration info
        """
        try:
            # Prepare retrieval statistics
            retrieval_stats = self._prepare_retrieval_stats(
                source_documents, content_type, generated_queries
            )
            
            # Prepare chat context
            chat_context = self._prepare_chat_context(chat_history)
            
            # Create interaction data
            interaction_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": {
                    "original_question": question,
                    "content_type": content_type,
                    "generated_queries": generated_queries
                },
                "retrieval": retrieval_stats,
                "response": {
                    "answer": answer
                },
                "performance": {
                    "processing_time": processing_time,
                    "tokens_used": None  # TODO: Add token usage if available
                },
                "chat_context": chat_context,
                "system_info": system_info
            }
            
            # Read existing logs
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Add new log
            logs.append(interaction_data)
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging interaction: {str(e)}")
    
    def _prepare_retrieval_stats(self, 
                                source_documents: List[Any], 
                                content_type: str,
                                generated_queries: List[str]) -> Dict[str, Any]:
        """Prepare retrieval statistics for logging.
        
        Args:
            source_documents: Retrieved documents
            content_type: The routing type
            generated_queries: Generated query variations
            
        Returns:
            Dictionary with retrieval statistics
        """
        # Count document types
        document_types = {
            "course": 0,
            "program": 0,
            "unknown": 0
        }
        
        documents_info = []
        for doc in source_documents:
            doc_type = doc.metadata.get("doc_type", "unknown")
            document_types[doc_type] = document_types.get(doc_type, 0) + 1
            
            documents_info.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "source": os.path.basename(doc.metadata.get("source", ""))
            })
        
        return {
            "total_documents": len(source_documents),
            "documents": documents_info,
            "document_types": document_types,
            "generated_queries": generated_queries,
            "routing_type": content_type
        }
    
    def _prepare_chat_context(self, chat_history: List[Any]) -> Dict[str, Any]:
        """Prepare chat context for logging.
        
        Args:
            chat_history: Chat memory messages
            
        Returns:
            Dictionary with chat context information
        """
        context_messages = []
        
        if chat_history:
            # Get last few messages for context
            recent_messages = chat_history[-6:]  # Last 6 messages (3 pairs)
            
            for msg in recent_messages:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    context_messages.append({
                        "role": msg.type,
                        "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                    })
        
        return {
            "chat_history": context_messages,
            "memory_window_size": Config.MEMORY_WINDOW_SIZE,
            "total_messages": len(chat_history) if chat_history else 0
        }
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent chat interactions.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of recent interactions
        """
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Return most recent interactions
            return logs[-limit:] if len(logs) > limit else logs
            
        except Exception as e:
            print(f"Error reading recent interactions: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged interactions.
        
        Returns:
            Dictionary with interaction statistics
        """
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                return {"total_interactions": 0}
            
            # Calculate statistics
            total_interactions = len(logs)
            content_types = {}
            avg_processing_time = 0
            
            for log in logs:
                # Count content types
                content_type = log.get("query", {}).get("content_type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                # Sum processing times
                processing_time = log.get("performance", {}).get("processing_time", 0)
                if processing_time:
                    avg_processing_time += processing_time
            
            # Calculate average processing time
            if total_interactions > 0:
                avg_processing_time = avg_processing_time / total_interactions
            
            return {
                "total_interactions": total_interactions,
                "content_type_distribution": content_types,
                "average_processing_time": avg_processing_time,
                "last_interaction": logs[-1].get("timestamp") if logs else None
            }
            
        except Exception as e:
            print(f"Error calculating stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_logs(self) -> bool:
        """Clear all logged interactions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.log_file, 'w') as f:
                json.dump([], f)
            return True
        except Exception as e:
            print(f"Error clearing logs: {str(e)}")
            return False
    
    def export_logs(self, output_file: str) -> bool:
        """Export logs to a different file.
        
        Args:
            output_file: Path to the output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            with open(output_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting logs: {str(e)}")
            return False 