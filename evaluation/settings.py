"""
Configuration settings for evaluation system.
"""

from eval_models import EvaluationConfig


def get_default_config() -> EvaluationConfig:
    """Get default evaluation configuration."""
    return EvaluationConfig(
        output_dir="data/evaluation",
        rouge_metrics=['rouge1', 'rouge2', 'rougeL'],
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        bert_scorer_lang="en",
        test_data_path="data/json/cse_merged_data.json"
    )


def get_fast_config() -> EvaluationConfig:
    """Get configuration optimized for speed (smaller models)."""
    return EvaluationConfig(
        output_dir="data/evaluation",
        rouge_metrics=['rouge1', 'rougeL'],  # Skip ROUGE-2 for speed
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        bert_scorer_lang="en",
        test_data_path="data/json/cse_merged_data.json"
    )


def get_comprehensive_config() -> EvaluationConfig:
    """Get configuration for comprehensive evaluation."""
    return EvaluationConfig(
        output_dir="data/evaluation",
        rouge_metrics=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        semantic_model="sentence-transformers/all-mpnet-base-v2",  # Better but slower model
        bert_scorer_lang="en",
        test_data_path="data/json/cse_merged_data.json"
    )