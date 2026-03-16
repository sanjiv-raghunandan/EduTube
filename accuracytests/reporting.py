import json
from pathlib import Path
from typing import Dict, List

class EvaluationReporter:
    """Handles result compilation and reporting."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent / "results"
        self.output_dir.mkdir(exist_ok=True)
    
    def compile_statistics(self, evaluation_results: Dict, results_detail: List[Dict]) -> Dict:
        """Compile evaluation statistics."""
        stats = {'total_tests': len(results_detail), 'metrics': {}}
        
        for metric_name, result in evaluation_results.items():
            if result is not None:
                try:
                    test_results = result if isinstance(result, list) else []
                    passed = sum(1 for tc in test_results if hasattr(tc, 'success') and tc.success)
                    failed = len(test_results) - passed
                    scores = [tc.score for tc in test_results if hasattr(tc, 'score') and tc.score is not None]
                    avg_score = sum(scores) / len(scores) if scores else 0.0
                    
                    stats['metrics'][metric_name] = {
                        'passed': passed,
                        'failed': failed,
                        'average_score': avg_score,
                        'total_evaluated': len(test_results)
                    }
                except Exception as e:
                    print(f"⚠️  Warning: Could not compile stats for {metric_name}: {e}")
                    stats['metrics'][metric_name] = None
            else:
                stats['metrics'][metric_name] = None
        
        return stats
    
    def print_summary(self, stats: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("📊 EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nTotal Test Cases: {stats['total_tests']}")
        print("\nMetric Results:")
        print("-" * 60)
        
        for metric_name, metric_stats in stats['metrics'].items():
            if metric_stats:
                print(f"\n{metric_name.upper()}:")
                print(f"  Passed: {metric_stats['passed']}")
                print(f"  Failed: {metric_stats['failed']}")
                print(f"  Average Score: {metric_stats['average_score']:.3f}")
            else:
                print(f"\n{metric_name.upper()}: Evaluation failed")
        
        print("\n" + "=" * 80)
    
    def print_local_summary(self, stats: Dict):
        """Print evaluation summary for local metrics."""
        print("\n" + "=" * 80)
        print("📊 LOCAL EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nTotal Test Cases: {stats['total_tests']}")
        print("\nMetric Results:")
        print("-" * 60)
        
        for metric_name, metric_stats in stats['metrics'].items():
            if metric_stats:
                print(f"\n{metric_name.upper()}:")
                print(f"  Average Score: {metric_stats['average_score']:.3f}")
                print(f"  Min Score: {metric_stats['min_score']:.3f}")
                print(f"  Max Score: {metric_stats['max_score']:.3f}")
                print(f"  Passed (≥0.7): {metric_stats['passed']}")
            else:
                print(f"\n{metric_name.upper()}: Evaluation failed")
        
        print("\n" + "=" * 80)
    
    def save_results(self, stats: Dict, results_detail: List[Dict]):
        """Save evaluation results to JSON files."""
        stats_file = self.output_dir / "evaluation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Statistics saved to: {stats_file}")
        
        details_file = self.output_dir / "evaluation_details.json"
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(results_detail, f, indent=2, ensure_ascii=False)
        print(f"💾 Detailed results saved to: {details_file}")