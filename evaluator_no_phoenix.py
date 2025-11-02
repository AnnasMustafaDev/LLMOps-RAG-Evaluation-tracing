"""
RAG Retrieval and Generation Evaluation System
WITHOUT Phoenix - Simplified version that always works
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from langchain_together import Together
from langchain.prompts import PromptTemplate
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    query: str
    retrieved_chunk: str
    relevance_score: float
    is_relevant: int  # 0 or 1
    ground_truth: Optional[int] = None


@dataclass
class EvaluationMetrics:
    """Data class for evaluation metrics"""
    precision: float
    recall: float
    accuracy: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    def to_dict(self):
        return asdict(self)


class RAGRetrievalEvaluator:
    """
    RAG Retrieval Evaluator using LLM-based relevance scoring
    """
    
    def __init__(self, together_api_key: str, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        """
        Initialize the RAG evaluator
        
        Args:
            together_api_key: Together AI API key
            model_name: Model to use for evaluation
        """
        self.llm = Together(
            model=model_name,
            together_api_key=together_api_key,
            temperature=0.0,
        )
        
        # Relevance evaluation prompt
        self.relevance_prompt = PromptTemplate(
            template="""You are an expert at evaluating document relevance for question answering.

Query: {query}

Retrieved Document: {document}

Task: Determine if the retrieved document is relevant to answering the query.
A document is relevant if it contains information that could help answer the query, even partially.

Respond with ONLY "1" if relevant or "0" if not relevant. No explanation needed.

Answer:""",
            input_variables=["query", "document"]
        )
        
        self.chain = self.relevance_prompt | self.llm
        
    def evaluate_chunk(self, query: str, chunk: str) -> Tuple[int, float]:
        """
        Evaluate a single retrieved chunk
        
        Args:
            query: The user query
            chunk: Retrieved document chunk
            
        Returns:
            Tuple of (binary_relevance, confidence_score)
        """
        try:
            response = self.chain.invoke({
                "query": query,
                "document": chunk
            })
            
            # Parse response
            response_text = response.strip()
            
            # Extract binary decision
            if "1" in response_text[:5]:
                binary_relevance = 1
                confidence = 0.9
            elif "0" in response_text[:5]:
                binary_relevance = 0
                confidence = 0.9
            else:
                logger.warning(f"Unclear response: {response_text}, defaulting to 0")
                binary_relevance = 0
                confidence = 0.5
                
            return binary_relevance, confidence
            
        except Exception as e:
            logger.error(f"Error evaluating chunk: {e}")
            return 0, 0.0
    
    def evaluate_retrieval_batch(
        self,
        queries: List[str],
        retrieved_chunks: List[List[str]],
        ground_truth_labels: Optional[List[List[int]]] = None
    ) -> List[RetrievalResult]:
        """
        Evaluate multiple queries and their retrieved chunks
        
        Args:
            queries: List of queries
            retrieved_chunks: List of lists of retrieved chunks per query
            ground_truth_labels: Optional ground truth relevance labels
            
        Returns:
            List of RetrievalResult objects
        """
        results = []
        
        for idx, (query, chunks) in enumerate(zip(queries, retrieved_chunks)):
            logger.info(f"Evaluating query {idx + 1}/{len(queries)}: {query[:50]}...")
            
            for chunk_idx, chunk in enumerate(chunks):
                binary_relevance, confidence = self.evaluate_chunk(query, chunk)
                
                ground_truth = None
                if ground_truth_labels and idx < len(ground_truth_labels):
                    if chunk_idx < len(ground_truth_labels[idx]):
                        ground_truth = ground_truth_labels[idx][chunk_idx]
                
                result = RetrievalResult(
                    query=query,
                    retrieved_chunk=chunk,
                    relevance_score=confidence,
                    is_relevant=binary_relevance,
                    ground_truth=ground_truth
                )
                results.append(result)
        
        return results
    
    def calculate_metrics(self, results: List[RetrievalResult]) -> EvaluationMetrics:
        """
        Calculate evaluation metrics from results
        
        Args:
            results: List of RetrievalResult objects with ground truth
            
        Returns:
            EvaluationMetrics object
        """
        # Filter results that have ground truth
        labeled_results = [r for r in results if r.ground_truth is not None]
        
        if not labeled_results:
            raise ValueError("No ground truth labels found in results")
        
        y_true = [r.ground_truth for r in labeled_results]
        y_pred = [r.is_relevant for r in labeled_results]
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1_score=f1,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn)
        )
        
        return metrics
    
    def print_classification_report(self, results: List[RetrievalResult]):
        """Print detailed classification report"""
        labeled_results = [r for r in results if r.ground_truth is not None]
        
        if not labeled_results:
            logger.warning("No ground truth labels found")
            return
        
        y_true = [r.ground_truth for r in labeled_results]
        y_pred = [r.is_relevant for r in labeled_results]
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            y_true, y_pred,
            target_names=["Not Relevant", "Relevant"],
            digits=4
        ))


def create_sample_data() -> Tuple[List[str], List[List[str]], List[List[int]]]:
    """
    Create sample data for demonstration
    
    Returns:
        Tuple of (queries, retrieved_chunks, ground_truth_labels)
    """
    queries = [
        "What is machine learning?",
        "How does a neural network work?",
        "What are the benefits of cloud computing?",
        "Explain quantum computing",
        "What is the capital of France?"
    ]
    
    retrieved_chunks = [
        [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "The Eiffel Tower is a famous landmark in Paris, France.",
            "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning."
        ],
        [
            "Neural networks are computing systems inspired by biological neural networks in animal brains.",
            "A neural network consists of interconnected nodes called neurons organized in layers.",
            "Pizza is a popular Italian dish made with dough, sauce, and cheese."
        ],
        [
            "Cloud computing provides on-demand access to computing resources over the internet.",
            "Benefits include scalability, cost-efficiency, and flexibility.",
            "The Great Wall of China is one of the world's most famous structures."
        ],
        [
            "Quantum computing uses quantum mechanical phenomena like superposition and entanglement.",
            "Quantum computers can solve certain problems exponentially faster than classical computers.",
            "Coffee is a popular beverage made from roasted coffee beans."
        ],
        [
            "Paris is the capital and largest city of France.",
            "France is located in Western Europe.",
            "The Amazon rainforest is the world's largest tropical rainforest."
        ]
    ]
    
    ground_truth_labels = [
        [1, 0, 1],  # For "What is machine learning?"
        [1, 1, 0],  # For "How does a neural network work?"
        [1, 1, 0],  # For "What are the benefits of cloud computing?"
        [1, 1, 0],  # For "Explain quantum computing"
        [1, 1, 0]   # For "What is the capital of France?"
    ]
    
    return queries, retrieved_chunks, ground_truth_labels


def main():
    """Main execution function"""
    try:
        # Get API key
        together_api_key = os.getenv("TOGETHER_API_KEY")
        if not together_api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        
        # Initialize components
        logger.info("Initializing RAG Evaluator...")
        evaluator = RAGRetrievalEvaluator(together_api_key)
        
        # Create sample data
        queries, retrieved_chunks, ground_truth_labels = create_sample_data()
        
        # Evaluate retrieval
        logger.info("Starting retrieval evaluation...")
        results = evaluator.evaluate_retrieval_batch(
            queries=queries,
            retrieved_chunks=retrieved_chunks,
            ground_truth_labels=ground_truth_labels
        )
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = evaluator.calculate_metrics(results)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Precision:  {metrics.precision:.4f}")
        print(f"Recall:     {metrics.recall:.4f}")
        print(f"Accuracy:   {metrics.accuracy:.4f}")
        print(f"F1 Score:   {metrics.f1_score:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"True Positives:  {metrics.true_positives}")
        print(f"False Positives: {metrics.false_positives}")
        print(f"True Negatives:  {metrics.true_negatives}")
        print(f"False Negatives: {metrics.false_negatives}")
        
        # Print classification report
        evaluator.print_classification_report(results)
        
        # Create results DataFrame
        logger.info("Creating results DataFrame...")
        data = []
        for result in results:
            data.append({
                "query": result.query,
                "retrieved_chunk": result.retrieved_chunk,
                "relevance_score": result.relevance_score,
                "is_relevant": result.is_relevant,
                "ground_truth": result.ground_truth
            })
        results_df = pd.DataFrame(data)
        
        # Save results
        results_df.to_csv("retrieval_evaluation_results.csv", index=False)
        logger.info("✅ Results saved to retrieval_evaluation_results.csv")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics.to_dict()])
        metrics_df.to_csv("evaluation_metrics.csv", index=False)
        logger.info("✅ Metrics saved to evaluation_metrics.csv")
        
        print("\n" + "="*60)
        print("✅ Evaluation complete!")
        print("📊 Results saved to CSV files")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()