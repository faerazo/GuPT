from rag_evaluation import evaluate
from rag_fe import RAGModel

# Initialize your RAG model
print("Initializing RAG model...")
rag_model = RAGModel(".")
print("Loading documents...")
rag_model.load_documents()

# Run evaluation on full dataset
print("Starting full evaluation...")
summary = evaluate(rag_model)

# Run evaluation on subset
# print("Starting evaluation on a subset of test cases...")
# summary = evaluate(rag_model, subset_size=3)

print("\nEvaluation complete!")
print("Results are saved in data/evaluation/ with timestamp in filenames:")
print("- eval_[timestamp].md: Human-readable summary")
print("- evaluation_results_[timestamp].jsonl: Detailed results for each test")
print("- aggregated_metrics_[timestamp].json: Overall metrics") 