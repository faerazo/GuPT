"""
Main evaluation system orchestrator.
"""

from datetime import datetime
from typing import Dict, Any, List
from dataclasses import asdict

from eval_models import ResponseMetrics, EvaluationConfig
from calculators import MetricsCalculator
from test_loader import TestCaseLoader
from output import OutputManager


class EvaluationSystem:
    """Main evaluation system that orchestrates all components."""
    
    def __init__(self, config: EvaluationConfig = None):
        """Initialize evaluation system with configuration."""
        if config is None:
            from settings import get_default_config
            config = get_default_config()
        
        self.config = config
        self.metrics_calculator = MetricsCalculator(config)
        self.test_loader = TestCaseLoader(config)
        self.output_manager = OutputManager(config.output_dir)
    
    def evaluate_response(self, query: str, response: str, ground_truth: str = None, 
                          contexts: List[str] = None, response_type: str = None,
                          model_version: str = None, embedding_version: str = None) -> ResponseMetrics:
        """Evaluate a single response."""
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

        # Compute evaluation metrics
        if ground_truth:
            metrics.rouge_scores = self.metrics_calculator.calculate_rouge_scores(response, ground_truth)
            metrics.semantic_similarity = self.metrics_calculator.calculate_semantic_similarity(response, ground_truth)
            metrics.bert_score = self.metrics_calculator.calculate_bert_score(response, ground_truth)

        if contexts:
            metrics.context_relevance = self.metrics_calculator.calculate_context_relevance(query, contexts)

        # Save individual result
        self.output_manager.save_individual_result(metrics)
        return metrics
    
    def run_test_suite(self, rag_model, subset_size: int = None, test_type: str = None) -> Dict[str, Any]:
        """Run the evaluation test suite."""
        all_results = []
        
        if test_type:
            # Run specific test type
            test_cases = self.test_loader.get_test_cases(test_type, subset_size)
            print(f"\nRunning {test_type} tests ({len(test_cases)} cases)...")
            all_results.extend(self._run_test_cases(rag_model, test_cases, test_type))
        else:
            # Run all test types
            for current_test_type in self.test_loader.get_test_types():
                test_cases = self.test_loader.get_test_cases(current_test_type, subset_size)
                print(f"\nRunning {current_test_type} tests ({len(test_cases)} cases)...")
                all_results.extend(self._run_test_cases(rag_model, test_cases, current_test_type))

        # Calculate and save aggregate metrics
        aggregate_metrics = self.metrics_calculator.calculate_aggregate_metrics(all_results)
        self.output_manager.save_aggregate_metrics(aggregate_metrics)
        self.output_manager.generate_summary_report(aggregate_metrics)
        self.output_manager.print_summary(aggregate_metrics)
        
        return aggregate_metrics
    
    def _run_test_cases(self, rag_model, test_cases, test_type: str) -> List[Dict]:
        """Run evaluation on a list of test cases."""
        results = []
        
        for i, case in enumerate(test_cases):
            try:
                print(f"  Processing {i+1}/{len(test_cases)}: {case.question[:50]}...")
                
                # Get response from RAG model
                response = rag_model.query(case.question)
                
                # Evaluate the response (QueryResult is a dataclass, not dict)
                metrics = self.evaluate_response(
                    query=case.question,
                    response=response.answer,
                    ground_truth=case.ground_truth,
                    contexts=[doc.page_content for doc in response.source_documents],
                    response_type=test_type,
                    model_version="gpt-4o-mini",
                    embedding_version="text-embedding-3-small"
                )
                
                results.append(asdict(metrics))
                
            except Exception as e:
                print(f"  Error evaluating case {i+1}: {e}")
                continue
        
        return results