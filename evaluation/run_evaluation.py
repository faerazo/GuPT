import argparse
import sys
import os

# Add parent, current, and src directories to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)      # For src modules (highest priority)
sys.path.insert(0, current_dir)  # For evaluation modules

from settings import get_default_config, get_fast_config, get_comprehensive_config
from rag_service import RAGService

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run RAG model evaluation')
    parser.add_argument('--subset', 
                       type=int, 
                       default=None,
                       help='Number of test cases to evaluate (default: all)')
    parser.add_argument('--config',
                       choices=['default', 'fast', 'comprehensive'],
                       default='default',
                       help='Evaluation configuration preset (default: default)')
    parser.add_argument('--test-type',
                       choices=['course_info', 'prerequisites', 'learning_outcomes', 'assessment'],
                       help='Run only specific test type (default: all)')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Get configuration
    if args.config == 'fast':
        config = get_fast_config()
        print("Using fast evaluation config...")
    elif args.config == 'comprehensive':
        config = get_comprehensive_config()
        print("Using comprehensive evaluation config...")
    else:
        config = get_default_config()
        print("Using default evaluation config...")

    # Initialize RAG service
    print("Initializing RAG service...")
    try:
        # Create RAG service (like main.py does)
        rag_service = RAGService()
        
        # Load documents (this is the missing step!)
        print("Loading documents into RAG service...")
        num_chunks = rag_service.load_documents()
        print(f"✅ RAG service initialized with {num_chunks} document chunks!")
        
    except Exception as e:
        print(f"Error initializing RAG service: {e}")
        print("Please ensure the RAG service is properly configured.")
        return

    # Run evaluation
    subset_msg = f" on a subset of {args.subset} test cases" if args.subset else ""
    test_type_msg = f" for {args.test_type} tests" if args.test_type else ""
    print(f"Starting evaluation{subset_msg}{test_type_msg}...")
    
    try:
        from evaluator import EvaluationSystem
        evaluator = EvaluationSystem(config)
        summary = evaluator.run_test_suite(rag_service, subset_size=args.subset, test_type=args.test_type)
        
        print("\nEvaluation complete!")
        print("Results are saved in data/evaluation/ with timestamp in filenames:")
        print("- eval_[timestamp].md: Human-readable summary")
        print("- evaluation_results_[timestamp].jsonl: Detailed results for each test")
        print("- aggregated_metrics_[timestamp].json: Overall metrics")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please check your configuration and data files.")

if __name__ == "__main__":
    main()