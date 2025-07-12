"""
Output and reporting utilities for evaluation results.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import asdict


class OutputManager:
    """Manages output files and reporting for evaluations."""
    
    def __init__(self, output_dir: str):
        """Initialize with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define output file paths
        self.results_file = self.output_dir / f"evaluation_results_{self.timestamp}.jsonl"
        self.metrics_file = self.output_dir / f"aggregated_metrics_{self.timestamp}.json"
        self.summary_file = self.output_dir / f"eval_{self.timestamp}.md"
    
    def save_individual_result(self, metrics) -> None:
        """Save individual evaluation result."""
        with open(self.results_file, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
    
    def save_aggregate_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save aggregated metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def generate_summary_report(self, aggregate_metrics: Dict[str, Any]) -> None:
        """Generate a human-readable summary report."""
        with open(self.summary_file, 'w') as f:
            f.write("# RAG Evaluation Summary\n\n")
            f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall metrics
            f.write("## Overall Metrics\n\n")
            f.write(f"- **Total Evaluations:** {aggregate_metrics.get('total_evaluations', 0)}\n")
            f.write(f"- **Valid Evaluations:** {aggregate_metrics.get('valid_evaluations', 0)}\n\n")
            
            # ROUGE scores
            f.write("### ROUGE Scores\n")
            rouge1 = aggregate_metrics.get('average_rouge1_f1')
            rouge2 = aggregate_metrics.get('average_rouge2_f1') 
            rougeL = aggregate_metrics.get('average_rougeL_f1')
            f.write(f"- **ROUGE-1 F1:** {f'{rouge1:.4f}' if rouge1 is not None else 'N/A'}\n")
            f.write(f"- **ROUGE-2 F1:** {f'{rouge2:.4f}' if rouge2 is not None else 'N/A'}\n")
            f.write(f"- **ROUGE-L F1:** {f'{rougeL:.4f}' if rougeL is not None else 'N/A'}\n\n")
            
            # Other metrics
            f.write("### Other Metrics\n")
            semantic = aggregate_metrics.get('average_semantic_similarity')
            f.write(f"- **Semantic Similarity:** {f'{semantic:.4f}' if semantic is not None else 'N/A'}\n")
            
            bert_score = aggregate_metrics.get('average_bert_score')
            f.write(f"- **BERT Score:** {f'{bert_score:.4f}' if bert_score else 'N/A'}\n")
            
            context_rel = aggregate_metrics.get('average_context_relevance')
            f.write(f"- **Context Relevance:** {f'{context_rel:.4f}' if context_rel else 'N/A'}\n\n")
            
            # By test type
            by_type = aggregate_metrics.get('by_test_type', {})
            if by_type:
                f.write("## Results by Test Type\n\n")
                for test_type, metrics in by_type.items():
                    f.write(f"### {test_type.replace('_', ' ').title()}\n")
                    f.write(f"- **Count:** {metrics.get('count', 0)}\n")
                    avg_rouge1 = metrics.get('avg_rouge1', 0)
                    avg_semantic = metrics.get('avg_semantic', 0)
                    f.write(f"- **Avg ROUGE-1:** {f'{avg_rouge1:.4f}' if avg_rouge1 else 'N/A'}\n")
                    f.write(f"- **Avg Semantic:** {f'{avg_semantic:.4f}' if avg_semantic else 'N/A'}\n\n")
            
            # Files generated
            f.write("## Files Generated\n\n")
            f.write(f"- **Detailed Results:** `{self.results_file.name}`\n")
            f.write(f"- **Aggregate Metrics:** `{self.metrics_file.name}`\n")
            f.write(f"- **Summary Report:** `{self.summary_file.name}`\n")
    
    def print_summary(self, aggregate_metrics: Dict[str, Any]) -> None:
        """Print evaluation summary to console."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Evaluations: {aggregate_metrics.get('total_evaluations', 0)}")
        print(f"Valid Evaluations: {aggregate_metrics.get('valid_evaluations', 0)}")
        rouge1 = aggregate_metrics.get('average_rouge1_f1')
        rouge2 = aggregate_metrics.get('average_rouge2_f1')
        rougeL = aggregate_metrics.get('average_rougeL_f1')
        semantic = aggregate_metrics.get('average_semantic_similarity')
        print(f"ROUGE-1 F1: {f'{rouge1:.4f}' if rouge1 is not None else 'N/A'}")
        print(f"ROUGE-2 F1: {f'{rouge2:.4f}' if rouge2 is not None else 'N/A'}")
        print(f"ROUGE-L F1: {f'{rougeL:.4f}' if rougeL is not None else 'N/A'}")
        print(f"Semantic Similarity: {f'{semantic:.4f}' if semantic is not None else 'N/A'}")
        
        bert_score = aggregate_metrics.get('average_bert_score')
        print(f"BERT Score: {f'{bert_score:.4f}' if bert_score else 'N/A'}")
        
        context_rel = aggregate_metrics.get('average_context_relevance')
        print(f"Context Relevance: {f'{context_rel:.4f}' if context_rel else 'N/A'}")
        
        print("="*60)
        print(f"Results saved to: {self.output_dir}")
        print("="*60)