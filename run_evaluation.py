import argparse
from rag_evaluation import evaluate
from rag import RAGModel

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run RAG model evaluation')
    parser.add_argument('--subset', 
                       type=int, 
                       default=None,
                       help='Number of test cases to evaluate (default: all)')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize your RAG model
    print("Initializing RAG model...")
    rag_model = RAGModel(".")
    print("Loading documents...")
    rag_model.load_documents()

    # Run evaluation
    if args.subset:
        print(f"Starting evaluation on a subset of {args.subset} test cases...")
        summary = evaluate(rag_model, subset_size=args.subset)
    else:
        print("Starting full evaluation...")
        summary = evaluate(rag_model)

    print("\nEvaluation complete!")
    print("Results are saved in data/evaluation/ with timestamp in filenames:")
    print("- eval_[timestamp].md: Human-readable summary")
    print("- evaluation_results_[timestamp].jsonl: Detailed results for each test")
    print("- aggregated_metrics_[timestamp].json: Overall metrics")

if __name__ == "__main__":
    main()