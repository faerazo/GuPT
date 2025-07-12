"""
Metrics calculation utilities.
"""

import numpy as np
from typing import Dict, List
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer


class MetricsCalculator:
    """Handles calculation of various evaluation metrics."""
    
    def __init__(self, config):
        """Initialize metrics calculator with configuration."""
        self.config = config
        self.rouge_scorer = rouge_scorer.RougeScorer(
            config.rouge_metrics, 
            use_stemmer=True
        )
        self.semantic_model = SentenceTransformer(config.semantic_model)
        self.bert_scorer = BERTScorer(
            lang=config.bert_scorer_lang, 
            rescale_with_baseline=False
        )
    
    def calculate_rouge_scores(self, response: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(ground_truth, response)
        return {
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rougeL_f1": scores["rougeL"].fmeasure
        }
    
    def calculate_semantic_similarity(self, response: str, ground_truth: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        response_embedding = self.semantic_model.encode([response])
        truth_embedding = self.semantic_model.encode([ground_truth])
        return float(np.dot(response_embedding, truth_embedding.T)[0][0])
    
    def calculate_bert_score(self, response: str, ground_truth: str) -> float:
        """Calculate BERTScore for the response compared to the ground truth."""
        try:
            P, R, F1 = self.bert_scorer.score([response], [ground_truth])
            return float(F1.mean())
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return None
    
    def calculate_context_relevance(self, query: str, contexts: List[str]) -> float:
        """Calculate relevance of retrieved contexts to the query."""
        if not contexts:
            return 0.0
        
        query_embedding = self.semantic_model.encode([query])
        context_embeddings = self.semantic_model.encode(contexts)
        similarities = np.dot(context_embeddings, query_embedding.T)
        return float(np.mean(similarities))
    
    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics from individual results."""
        if not results:
            return {}
        
        # Filter results that have the required metrics
        valid_results = [r for r in results if r.get('rouge_scores') and r.get('semantic_similarity')]
        
        if not valid_results:
            return {"error": "No valid results found"}
        
        # Calculate averages
        avg_rouge1 = np.mean([r['rouge_scores']['rouge1_f1'] for r in valid_results])
        avg_rouge2 = np.mean([r['rouge_scores']['rouge2_f1'] for r in valid_results])
        avg_rougeL = np.mean([r['rouge_scores']['rougeL_f1'] for r in valid_results])
        avg_semantic = np.mean([r['semantic_similarity'] for r in valid_results])
        
        # BERTScore (may be None for some results)
        bert_scores = [r.get('bert_score') for r in valid_results if r.get('bert_score') is not None]
        avg_bert = np.mean(bert_scores) if bert_scores else None
        
        # Context relevance (may be None for some results)
        context_relevances = [r.get('context_relevance') for r in valid_results if r.get('context_relevance') is not None]
        avg_context_relevance = np.mean(context_relevances) if context_relevances else None
        
        return {
            "total_evaluations": len(results),
            "valid_evaluations": len(valid_results),
            "average_rouge1_f1": float(avg_rouge1),
            "average_rouge2_f1": float(avg_rouge2),
            "average_rougeL_f1": float(avg_rougeL),
            "average_semantic_similarity": float(avg_semantic),
            "average_bert_score": float(avg_bert) if avg_bert is not None else None,
            "average_context_relevance": float(avg_context_relevance) if avg_context_relevance is not None else None,
            "by_test_type": self._calculate_by_test_type(valid_results)
        }
    
    def _calculate_by_test_type(self, results: List[Dict]) -> Dict:
        """Calculate metrics grouped by test type."""
        by_type = {}
        
        for result in results:
            test_type = result.get('response_type', 'unknown')
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(result)
        
        type_metrics = {}
        for test_type, type_results in by_type.items():
            if type_results:
                type_metrics[test_type] = {
                    "count": len(type_results),
                    "avg_rouge1": np.mean([r['rouge_scores']['rouge1_f1'] for r in type_results]),
                    "avg_semantic": np.mean([r['semantic_similarity'] for r in type_results])
                }
        
        return type_metrics