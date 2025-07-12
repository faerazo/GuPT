import os
import re
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import Config, AppConstants
from models import DocumentMetadata, ProcessingStats

class DocumentProcessor:
    """Handles document loading, processing, and chunking."""
    
    def __init__(self, base_path: str = None):
        """Initialize the document processor.
        
        Args:
            base_path: Base path for document directories
        """
        self.base_path = base_path or Config.DATA_BASE_PATH
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_all_documents(self) -> List[Document]:
        """Process both markdown and PDF documents from courses and programs directories.
        
        Returns:
            List of processed documents with proper metadata
        """
        start_time = time.time()
        
        documents = {
            'courses': [],
            'programs': []
        }
        
        # Define paths for different document types
        paths = self._get_document_paths()
        
        # Create directories if they don't exist
        self._ensure_directories_exist(paths)
        
        # Process documents by category
        for category in ['courses', 'programs']:
            # Process markdown files
            md_path = paths[f'{category}_md']
            if os.path.exists(md_path):
                documents[category].extend(self._process_markdown_files(md_path, category))
            
            # Process PDF files
            pdf_path = paths[f'{category}_pdf']
            if os.path.exists(pdf_path):
                documents[category].extend(self._process_pdf_files(pdf_path, category))
            
            print(f"Processed {len(documents[category])} {category} documents")
        
        # Combine all documents while maintaining their metadata
        all_documents = documents['courses'] + documents['programs']
        
        # Create processing stats
        processing_time = time.time() - start_time
        stats = ProcessingStats(
            total_documents=len(all_documents),
            courses_processed=len(documents['courses']),
            programs_processed=len(documents['programs']),
            chunks_created=0,  # Will be updated after chunking
            processing_time=processing_time
        )
        
        print(f"Total documents processed: {len(all_documents)}")
        print(f"Courses: {len(documents['courses'])}, Programs: {len(documents['programs'])}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for embedding.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks
        """
        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} document chunks")
        return chunks
    
    def _get_document_paths(self) -> Dict[str, str]:
        """Get paths for different document types.
        
        Returns:
            Dictionary with document paths
        """
        return {
            'courses_md': os.path.join(self.base_path, Config.COURSES_MD_PATH),
            'courses_pdf': os.path.join(self.base_path, Config.COURSES_PDF_PATH),
            'programs_md': os.path.join(self.base_path, Config.PROGRAMS_MD_PATH),
            'programs_pdf': os.path.join(self.base_path, Config.PROGRAMS_PDF_PATH)
        }
    
    def _ensure_directories_exist(self, paths: Dict[str, str]) -> None:
        """Ensure all document directories exist.
        
        Args:
            paths: Dictionary of paths to create
        """
        for path in paths.values():
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")
    
    def _process_markdown_files(self, path: str, category: str) -> List[Document]:
        """Process markdown files in a directory.
        
        Args:
            path: Path to the markdown files directory
            category: Type of documents ('courses' or 'programs')
            
        Returns:
            List of processed markdown documents with metadata
        """
        documents = []
        
        if not os.path.exists(path):
            print(f"Warning: Markdown directory {path} does not exist")
            return documents
        
        for filename in os.listdir(path):
            if filename.endswith('.md'):
                file_path = os.path.join(path, filename)
                try:
                    content = self._read_file_with_fallback_encoding(file_path)
                    
                    # Create metadata
                    metadata = {
                        'source': file_path,
                        'type': 'markdown',
                        'category': category,
                        'doc_type': category.rstrip('s'),  # 'course' or 'program'
                        'filename': filename
                    }
                    
                    # Extract course code if it's a course document
                    if category == 'courses':
                        code = self._extract_course_code(filename, content)
                        if code:
                            metadata['course_code'] = code
                    
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    print(f"Error processing markdown file {filename}: {str(e)}")
        
        return documents
    
    def _process_pdf_files(self, path: str, category: str) -> List[Document]:
        """Process PDF files in a directory.
        
        Args:
            path: Path to the PDF files directory
            category: Type of documents ('courses' or 'programs')
            
        Returns:
            List of processed PDF documents with metadata
        """
        documents = []
        
        if not os.path.exists(path):
            print(f"Warning: PDF directory {path} does not exist")
            return documents
        
        for filename in os.listdir(path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(path, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    pdf_docs = loader.load()
                    
                    # Create base metadata
                    metadata = {
                        'type': 'pdf',
                        'category': category,
                        'doc_type': category.rstrip('s'),  # 'course' or 'program'
                        'filename': filename
                    }
                    
                    # Add course code if it exists and it's a course document
                    if category == 'courses' and pdf_docs:
                        code = self._extract_course_code(filename, pdf_docs[0].page_content)
                        if code:
                            metadata['course_code'] = code
                    
                    # Add metadata to each page
                    for doc in pdf_docs:
                        doc.metadata.update(metadata)
                    
                    documents.extend(pdf_docs)
                    
                except Exception as e:
                    print(f"Error processing PDF {filename}: {str(e)}")
        
        return documents
    
    def _read_file_with_fallback_encoding(self, file_path: str) -> str:
        """Read a file with fallback encodings.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content as string
            
        Raises:
            UnicodeDecodeError: If file cannot be read with any encoding
        """
        for encoding in AppConstants.SUPPORTED_FILE_ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError(f"Failed to read {file_path} with any encoding")
    
    def _extract_course_code(self, filename: str, content: str) -> Optional[str]:
        """Extract course code from filename or content if possible.
        
        Args:
            filename: Name of the file
            content: Content of the document
            
        Returns:
            Course code if found, None otherwise
        """
        # Try to extract from filename first (e.g., "DIT134-advanced-programming.pdf")
        code_match = re.search(r'([A-Z]{3}\d{3})', filename)
        if code_match:
            return code_match.group(1)
        
        # Try to extract from content (first occurrence)
        code_match = re.search(r'([A-Z]{3}\d{3})', content[:1000])  # Search in first 1000 chars
        if code_match:
            return code_match.group(1)
        
        return None
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed documents.
        
        Args:
            documents: List of processed documents
            
        Returns:
            Dictionary with document statistics
        """
        stats = {
            'total_documents': len(documents),
            'by_category': {},
            'by_type': {},
            'by_doc_type': {},
            'course_codes': set(),
            'total_content_length': 0
        }
        
        for doc in documents:
            metadata = doc.metadata
            
            # Count by category
            category = metadata.get('category', 'unknown')
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
            
            # Count by file type
            file_type = metadata.get('type', 'unknown')
            stats['by_type'][file_type] = stats['by_type'].get(file_type, 0) + 1
            
            # Count by document type
            doc_type = metadata.get('doc_type', 'unknown')
            stats['by_doc_type'][doc_type] = stats['by_doc_type'].get(doc_type, 0) + 1
            
            # Collect course codes
            if metadata.get('course_code'):
                stats['course_codes'].add(metadata['course_code'])
            
            # Sum content length
            stats['total_content_length'] += len(doc.page_content)
        
        # Convert set to list for JSON serialization
        stats['course_codes'] = list(stats['course_codes'])
        stats['unique_course_codes'] = len(stats['course_codes'])
        
        return stats
    
    def validate_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Validate processed documents for common issues.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_documents': len(documents),
            'issues': [],
            'warnings': [],
            'valid_documents': 0,
            'empty_documents': 0,
            'missing_metadata': 0
        }
        
        for i, doc in enumerate(documents):
            # Check for empty content
            if not doc.page_content or len(doc.page_content.strip()) == 0:
                validation_results['empty_documents'] += 1
                validation_results['issues'].append(f"Document {i}: Empty content")
                continue
            
            # Check for essential metadata
            required_metadata = ['source', 'type', 'category', 'doc_type', 'filename']
            missing_fields = [field for field in required_metadata if not doc.metadata.get(field)]
            
            if missing_fields:
                validation_results['missing_metadata'] += 1
                validation_results['warnings'].append(
                    f"Document {i}: Missing metadata fields: {missing_fields}"
                )
            
            # Check content length
            if len(doc.page_content) < 50:
                validation_results['warnings'].append(
                    f"Document {i}: Very short content ({len(doc.page_content)} chars)"
                )
            
            validation_results['valid_documents'] += 1
        
        return validation_results 