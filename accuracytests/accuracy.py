from config import (
    DEFAULT_DATASET_PATH, DEFAULT_MODEL_NAME, DEFAULT_DB_DIR,
    DEFAULT_CLEAN_DIR, DEFAULT_N_RESULTS
)
from evaluation import EduTubeEvaluator

def main():
    """Main entry point for running accuracy tests."""
    evaluator = EduTubeEvaluator(
        dataset_path=DEFAULT_DATASET_PATH,
        model_name=DEFAULT_MODEL_NAME,
        db_dir=DEFAULT_DB_DIR,
        clean_dir=DEFAULT_CLEAN_DIR,
        n_results=DEFAULT_N_RESULTS
    )
    
    results = evaluator.run_evaluation(limit=None)
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()