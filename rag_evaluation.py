import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field

@dataclass
class ResponseMetrics:
    timestamp: str
    query: str
    response: str
    ground_truth: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    rouge_scores: Optional[Dict[str, float]] = None
    semantic_similarity: Optional[float] = None
    context_relevance: Optional[float] = None
    confidence_score: Optional[float] = None
    response_type: Optional[str] = None
    model_version: Optional[str] = None
    embedding_version: Optional[str] = None

class EvaluationSystem:
    def __init__(self, output_dir: str = "data/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize semantic similarity model
        self.semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load test cases
        self.test_cases = self.load_test_cases()
        
        # Initialize results storage
        self.results_file = self.output_dir / "evaluation_results.jsonl"
        self.metrics_file = self.output_dir / "aggregated_metrics.json"

    def load_test_cases(self) -> Dict[str, List[Dict]]:
        """Load test cases from merged_data.json"""
        with open("data/json/merged_data.json", 'r') as f:
            courses_data = json.load(f)

        test_cases = {
            "course_info": [],
            "prerequisites": [],
            "learning_outcomes": [],
            "assessment": []
        }

        for course in courses_data:
            # Course information test cases
            test_cases["course_info"].append({
                "question": f"What is the {course['course_name']} ({course['course_code']}) course about?",
                "ground_truth": course['course_content'],
                "type": "course_info"
            })

            # Prerequisites test cases
            test_cases["prerequisites"].append({
                "question": f"What are the prerequisites for {course['course_code']}?",
                "ground_truth": course['entry_requirements'],
                "type": "prerequisites"
            })

            # Learning outcomes test cases
            if course.get('learning_outcomes'):
                outcomes = course['learning_outcomes'][0]
                test_cases["learning_outcomes"].append({
                    "question": f"What are the learning outcomes for {course['course_code']}?",
                    "ground_truth": str(outcomes),
                    "type": "learning_outcomes"
                })

            # Assessment test cases
            test_cases["assessment"].append({
                "question": f"How is {course['course_code']} assessed?",
                "ground_truth": course['assessment'],
                "type": "assessment"
            })

        return test_cases

    def calculate_rouge_scores(self, response: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(ground_truth, response)
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_f1': scores['rougeL'].fmeasure
        }

    def calculate_semantic_similarity(self, response: str, ground_truth: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        response_embedding = self.semantic_model.encode([response])
        truth_embedding = self.semantic_model.encode([ground_truth])
        return float(np.dot(response_embedding, truth_embedding.T)[0][0])

    def calculate_context_relevance(self, query: str, contexts: List[str]) -> float:
        """Calculate relevance of retrieved contexts to the query"""
        if not contexts:
            return 0.0
        
        query_embedding = self.semantic_model.encode([query])
        context_embeddings = self.semantic_model.encode(contexts)
        similarities = np.dot(context_embeddings, query_embedding.T)
        return float(np.mean(similarities))

    def evaluate_response(self, query: str, response: str, ground_truth: str = None, 
                         contexts: List[str] = None, response_type: str = None,
                         model_version: str = None, embedding_version: str = None) -> ResponseMetrics:
        """Evaluate a single response"""
        metrics = ResponseMetrics(
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response,
            ground_truth=ground_truth,
            retrieved_contexts=contexts,
            response_type=response_type,
            model_version=model_version,
            embedding_version=embedding_version
        )

        if ground_truth:
            metrics.rouge_scores = self.calculate_rouge_scores(response, ground_truth)
            metrics.semantic_similarity = self.calculate_semantic_similarity(response, ground_truth)

        if contexts:
            metrics.context_relevance = self.calculate_context_relevance(query, contexts)

        # Save individual result
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')

        return metrics

    def run_test_suite(self, rag_model) -> Dict[str, Any]:
        """Run the full test suite"""
        all_results = []
        
        for test_type, cases in self.test_cases.items():
            print(f"Running {test_type} tests...")
            for case in cases:
                try:
                    response = rag_model.query(case['question'])
                    metrics = self.evaluate_response(
                        query=case['question'],
                        response=response['answer'],
                        ground_truth=case['ground_truth'],
                        contexts=[doc.page_content for doc in response['source_documents']],
                        response_type=test_type,
                        model_version="gpt-4o-mini", # do not change this
                        embedding_version="text-embedding-3-small"
                    )
                    all_results.append(asdict(metrics))
                except Exception as e:
                    print(f"Error evaluating {case['question']}: {str(e)}")

        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(all_results)
        
        # Save aggregate metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)

        return aggregate_metrics

    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics from all results"""
        df = pd.DataFrame(results)
        
        metrics = {
            "total_queries": len(results),
            "average_metrics": {
                "semantic_similarity": df['semantic_similarity'].mean(),
                "context_relevance": df['context_relevance'].mean()
            },
            "by_type": {},
            "confidence_intervals": {}
        }

        # Calculate metrics by response type
        for response_type in df['response_type'].unique():
            type_df = df[df['response_type'] == response_type]
            metrics["by_type"][response_type] = {
                "count": len(type_df),
                "avg_semantic_similarity": type_df['semantic_similarity'].mean(),
                "avg_context_relevance": type_df['context_relevance'].mean()
            }

        # Calculate confidence intervals
        for metric in ['semantic_similarity', 'context_relevance']:
            mean = df[metric].mean()
            std = df[metric].std()
            conf_int = 1.96 * (std / np.sqrt(len(df)))  # 95% confidence interval
            metrics["confidence_intervals"][metric] = {
                "mean": float(mean),
                "lower_bound": float(mean - conf_int),
                "upper_bound": float(mean + conf_int)
            }

        return metrics

    def get_evaluation_summary(self) -> str:
        """Generate a human-readable summary of the evaluation results"""
        if not self.metrics_file.exists():
            return """
### Evaluation Summary
No evaluation results available yet. Please run the evaluation first using the 'Run Full Evaluation' button.
"""
            
        try:
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)

            summary = f"""
### Evaluation Summary

#### Overall Statistics
- Total Queries Evaluated: {metrics['total_queries']}
- Average Semantic Similarity: {metrics['average_metrics']['semantic_similarity']:.3f}
- Average Context Relevance: {metrics['average_metrics']['context_relevance']:.3f}

#### Performance by Query Type:
"""
            for qtype, stats in metrics['by_type'].items():
                summary += f"""
- {qtype.replace('_', ' ').title()}:
  - Count: {stats['count']}
  - Avg Semantic Similarity: {stats['avg_semantic_similarity']:.3f}
  - Avg Context Relevance: {stats['avg_context_relevance']:.3f}
"""

            summary += "\n#### Confidence Intervals (95%):"
            for metric, ci in metrics['confidence_intervals'].items():
                summary += f"""
- {metric.replace('_', ' ').title()}:
  - Mean: {ci['mean']:.3f}
  - Range: [{ci['lower_bound']:.3f}, {ci['upper_bound']:.3f}]
"""
            return summary
        except Exception as e:
            return f"Error generating summary: {str(e)}" 