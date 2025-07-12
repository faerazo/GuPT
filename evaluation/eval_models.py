"""
Data models for evaluation system.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ResponseMetrics:
    """Metrics for evaluating a single response."""
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
    bert_score: Optional[float] = None
    academic_accuracy: Optional[float] = None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation system."""
    output_dir: str = "data/evaluation"
    rouge_metrics: List[str] = None
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bert_scorer_lang: str = "en"
    test_data_path: str = "data/json/cse_merged_data.json"
    
    def __post_init__(self):
        if self.rouge_metrics is None:
            self.rouge_metrics = ['rouge1', 'rouge2', 'rougeL']


@dataclass
class TestCase:
    """Represents a single test case."""
    question: str
    ground_truth: str
    test_type: str
    course_code: Optional[str] = None
    course_name: Optional[str] = None