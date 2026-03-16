import sys
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from assistant import EduTubeAssistant
from deepeval.test_case import LLMTestCase
from metrics import initialize_metrics
from reporting import EvaluationReporter


class EduTubeEvaluator:
    """Core evaluation engine with local metrics."""
    
    def __init__(self, dataset_path: str, model_name: str, db_dir: str, clean_dir: str, n_results: int):
        print("🔄 Initializing EduTube Assistant...")
        self.assistant = EduTubeAssistant(model_name=model_name, db_dir=db_dir, clean_dir=clean_dir)
        print("✅ Assistant initialized\n")
        
        self.dataset_path = Path(dataset_path)
        self.n_results = n_results
        self.local_metrics = initialize_metrics()  # Now returns LocalMetrics instance
        self.reporter = EvaluationReporter()
    
    def load_golden_dataset(self) -> List[Dict]:
        """Load golden dataset from JSONL file."""
        print(f"🔄 Loading golden dataset from {self.dataset_path}...")
        
        test_cases = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    messages = data.get('messages', [])
                    
                    if len(messages) >= 2:
                        user_msg = next((m for m in messages if m['role'] == 'user'), None)
                        assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
                        
                        if user_msg and assistant_msg:
                            test_cases.append({
                                'question': user_msg['content'],
                                'expected_answer': assistant_msg['content']
                            })
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Skipping malformed line {line_num}: {e}")
        
        print(f"✅ Loaded {len(test_cases)} test cases\n")
        return test_cases
    
    def generate_response(self, question: str) -> Dict:
        """Generate response using the assistant with RAG context."""
        try:
            result = self.assistant.query_with_context(
                question=question,
                n_results=self.n_results,
                temperature=0.7,
                max_tokens=2000
            )
            
            return {
                'answer': result['answer'],
                'context': [hit['text'] for hit in result['sources']],
                'sources': result['sources']
            }
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return {'answer': f"Error: {str(e)}", 'context': [], 'sources': []}
    
    def create_test_case(self, question: str, expected_answer: str, actual_output: Dict) -> LLMTestCase:
        """Create a deepeval LLMTestCase."""
        return LLMTestCase(
            input=question,
            actual_output=actual_output['answer'],
            expected_output=expected_answer,
            retrieval_context=actual_output['context']
        )
    
    def run_evaluation(self, limit: int = None) -> Dict:
        """Run full evaluation suite on golden dataset."""
        print("=" * 80)
        print("🧪 STARTING EDUTUBE ACCURACY EVALUATION (LOCAL METRICS)")
        print("=" * 80 + "\n")
        
        golden_dataset = self.load_golden_dataset()
        
        if limit:
            golden_dataset = golden_dataset[:limit]
            print(f"⚠️  Limiting evaluation to first {limit} test cases\n")
        
        print("🔄 Generating responses with RAG context...")
        test_cases = []
        results_detail = []
        evaluation_results = []
        
        for i, test_data in enumerate(tqdm(golden_dataset, desc="Processing"), 1):
            question = test_data['question']
            expected = test_data['expected_answer']
            
            actual_output = self.generate_response(question)
            test_case = self.create_test_case(question, expected, actual_output)
            test_cases.append(test_case)
            
            # Evaluate locally
            scores = self.local_metrics.evaluate_test_case(test_case)
            
            results_detail.append({
                'test_id': i,
                'question': question,
                'expected_answer': expected,
                'actual_answer': actual_output['answer'],
                'num_contexts': len(actual_output['context']),
                'sources': [s.get('video_title', 'Unknown') for s in actual_output['sources']],
                'scores': scores
            })
            
            evaluation_results.append(scores)
        
        print(f"✅ Generated and evaluated {len(test_cases)} responses\n")
        
        # Calculate summary statistics
        stats = self._calculate_local_stats(evaluation_results, results_detail)
        self.reporter.print_local_summary(stats)
        self.reporter.save_results(stats, results_detail)
        
        return {'statistics': stats, 'details': results_detail}
    
    def _calculate_local_stats(self, evaluation_results: List[Dict], results_detail: List[Dict]) -> Dict:
        """Calculate statistics from local evaluation results."""
        if not evaluation_results:
            return {'total_tests': 0, 'metrics': {}}
        
        # Calculate averages for each metric
        semantic_scores = [r['semantic_similarity'] for r in evaluation_results]
        context_scores = [r['context_relevance'] for r in evaluation_results]
        answer_scores = [r['answer_relevance'] for r in evaluation_results]
        
        return {
            'total_tests': len(results_detail),
            'metrics': {
                'semantic_similarity': {
                    'average_score': np.mean(semantic_scores),
                    'min_score': np.min(semantic_scores),
                    'max_score': np.max(semantic_scores),
                    'passed': sum(1 for s in semantic_scores if s >= 0.7)
                },
                'context_relevance': {
                    'average_score': np.mean(context_scores),
                    'min_score': np.min(context_scores),
                    'max_score': np.max(context_scores),
                    'passed': sum(1 for s in context_scores if s >= 0.7)
                },
                'answer_relevance': {
                    'average_score': np.mean(answer_scores),
                    'min_score': np.min(answer_scores),
                    'max_score': np.max(answer_scores),
                    'passed': sum(1 for s in answer_scores if s >= 0.7)
                }
            }
        }