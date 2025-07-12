#!/usr/bin/env python3
"""
GuPT: Gothenburg University Information Assistant
Main entry point for the restructured RAG application.

This is the modernized version using:
- LCEL (LangChain Expression Language)
- Modular architecture
- Better error handling
- Enhanced logging
"""

import sys
import time
import argparse
from typing import Optional

# Local imports
from config import Config, validate_config
from rag_service import RAGService
from interface import create_interface

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GuPT: Gothenburg University Information Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Launch with default settings
  python main.py --no-share         # Launch without sharing
  python main.py --port 8080        # Launch on port 8080
  python main.py --rebuild-db       # Force rebuild of vector database
        """
    )
    
    # Interface options
    parser.add_argument(
        "--share", 
        action="store_true", 
        default=False,
        help="Share the interface via Gradio public link"
    )
    parser.add_argument(
        "--no-share", 
        action="store_true", 
        default=False,
        help="Explicitly disable sharing (default)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="Port to run the interface on (default: 7860)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    # Database options
    parser.add_argument(
        "--rebuild-db", 
        action="store_true",
        help="Force rebuild of the vector database"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help=f"Custom path for vector database (default: {Config.CHROMA_DB_PATH})"
    )
    
    # Debug options
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with verbose output"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress non-essential output"
    )
    
    return parser.parse_args()

def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🎓 GuPT - Gothenburg University Information Assistant      ║
║                                                              ║
║   Built with: LangChain + OpenAI + Gradio                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_prerequisites() -> bool:
    """Check if all prerequisites are met.
    
    Returns:
        True if all prerequisites are met, False otherwise
    """
    try:
        # Validate configuration
        validate_config()
        print("✅ Configuration validated")
        
        # Check if required directories exist
        import os
        data_dirs = [
            Config.COURSES_MD_PATH,
            Config.COURSES_PDF_PATH,
            Config.PROGRAMS_MD_PATH,
            Config.PROGRAMS_PDF_PATH
        ]
        
        missing_dirs = []
        for dir_path in data_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print("⚠️  Warning: Some data directories are missing:")
            for dir_path in missing_dirs:
                print(f"   - {dir_path}")
            print("   The system will create them automatically if needed.")
        
        print("✅ Prerequisites check completed")
        return True
        
    except Exception as e:
        print(f"❌ Prerequisites check failed: {str(e)}")
        return False

def initialize_rag_service(args) -> Optional[RAGService]:
    """Initialize the RAG service.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Initialized RAG service or None if failed
    """
    try:
        print("🔧 Initializing RAG service...")
        
        # Create RAG service
        rag_service = RAGService()
        
        print("📚 Loading documents and vector store...")
        start_time = time.time()
        
        # Handle database rebuild
        if args.rebuild_db:
            print("🔄 Rebuilding vector database...")
            import shutil
            import os
            if os.path.exists(Config.CHROMA_DB_PATH):
                shutil.rmtree(Config.CHROMA_DB_PATH)
                print(f"   Removed existing database at {Config.CHROMA_DB_PATH}")
        
        # Load documents
        num_chunks = rag_service.load_documents()
        load_time = time.time() - start_time
        
        print(f"✅ RAG service initialized successfully!")
        print(f"   📊 Processed {num_chunks} document chunks")
        print(f"   ⏱️  Loading time: {load_time:.2f} seconds")
        
        return rag_service
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG service: {str(e)}")
        return None

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up quiet mode
    if args.quiet:
        import os
        # Redirect stdout to devnull for quiet mode
        # We'll still print important messages to stderr
        pass
    
    # Print banner unless in quiet mode
    if not args.quiet:
        print_banner()
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            print("❌ Prerequisites check failed. Please fix the issues and try again.")
            sys.exit(1)
        
        # Initialize RAG service
        rag_service = initialize_rag_service(args)
        if not rag_service:
            print("❌ Failed to initialize RAG service. Exiting.")
            sys.exit(1)
        
        # Create and launch interface
        print("🚀 Creating Gradio interface...")
        interface_wrapper = create_interface(rag_service)
        
        # Determine share setting
        share = args.share and not args.no_share
        
        # Launch parameters
        launch_params = {
            "share": share,
            "server_name": args.host,
            "server_port": args.port,
            "show_error": True,
            "quiet": args.quiet
        }
        
        print(f"🌐 Launching interface...")
        if not args.quiet:
            print(f"   📍 Local URL: http://{args.host}:{args.port}")
            if share:
                print(f"   🌍 Public sharing: Enabled")
            else:
                print(f"   🔒 Public sharing: Disabled")
        
        # Launch the interface
        interface_wrapper.create_interface()
        interface_wrapper.launch(**launch_params)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 