
from evaluator import RAGEvaluator
from config import Config

def main():
    config = Config()
    evaluator = RAGEvaluator(config)
    # Example usage:
    # evaluator.run_evaluation(...)

if __name__ == "__main__":
    main()
