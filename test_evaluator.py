"""
Unit tests for RAG evaluation system
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluator import (
    RetrievalResult,
    EvaluationMetrics,
    RAGRetrievalEvaluator,
    create_sample_data
)
from utils import (
    calculate_metrics_by_query,
    analyze_error_patterns
)
import pandas as pd


class TestRetrievalResult:
    """Test RetrievalResult dataclass"""
    
    def test_creation(self):
        result = RetrievalResult(
            query="test query",
            retrieved_chunk="test chunk",
            relevance_score=0.9,
            is_relevant=1,
            ground_truth=1
        )
        assert result.query == "test query"
        assert result.is_relevant == 1
        assert result.ground_truth == 1


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass"""
    
    def test_metrics_creation(self):
        metrics = EvaluationMetrics(
            precision=0.8,
            recall=0.9,
            accuracy=0.85,
            f1_score=0.845,
            true_positives=8,
            false_positives=2,
            true_negatives=7,
            false_negatives=1
        )
        assert metrics.precision == 0.8
        assert metrics.f1_score == 0.845
    
    def test_to_dict(self):
        metrics = EvaluationMetrics(
            precision=0.8,
            recall=0.9,
            accuracy=0.85,
            f1_score=0.845,
            true_positives=8,
            false_positives=2,
            true_negatives=7,
            false_negatives=1
        )
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['precision'] == 0.8
        assert metrics_dict['true_positives'] == 8


class TestRAGRetrievalEvaluator:
    """Test RAGRetrievalEvaluator class"""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        mock = Mock()
        mock.invoke = Mock(return_value="1")
        return mock
    
    @pytest.fixture
    def evaluator(self, mock_llm):
        """Create evaluator with mocked LLM"""
        with patch('evaluator.Together', return_value=mock_llm):
            evaluator = RAGRetrievalEvaluator(together_api_key="test_key")
            evaluator.llm = mock_llm
            return evaluator
    
    def test_evaluate_chunk_relevant(self, evaluator, mock_llm):
        """Test chunk evaluation for relevant document"""
        mock_llm.invoke = Mock(return_value="1")
        
        query = "What is AI?"
        chunk = "AI stands for Artificial Intelligence"
        
        binary, confidence = evaluator.evaluate_chunk(query, chunk)
        
        assert binary == 1
        assert confidence > 0
    
    def test_evaluate_chunk_not_relevant(self, evaluator, mock_llm):
        """Test chunk evaluation for non-relevant document"""
        mock_llm.invoke = Mock(return_value="0")
        
        query = "What is AI?"
        chunk = "Pizza is delicious food"
        
        binary, confidence = evaluator.evaluate_chunk(query, chunk)
        
        assert binary == 0
    
    def test_calculate_metrics_perfect_prediction(self, evaluator):
        """Test metrics calculation with perfect predictions"""
        results = [
            RetrievalResult("q1", "c1", 0.9, 1, 1),
            RetrievalResult("q1", "c2", 0.1, 0, 0),
            RetrievalResult("q2", "c3", 0.95, 1, 1),
            RetrievalResult("q2", "c4", 0.05, 0, 0),
        ]
        
        metrics = evaluator.calculate_metrics(results)
        
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.f1_score == 1.0
    
    def test_calculate_metrics_mixed_prediction(self, evaluator):
        """Test metrics calculation with mixed predictions"""
        results = [
            RetrievalResult("q1", "c1", 0.9, 1, 1),  # TP
            RetrievalResult("q1", "c2", 0.8, 1, 0),  # FP
            RetrievalResult("q2", "c3", 0.2, 0, 1),  # FN
            RetrievalResult("q2", "c4", 0.1, 0, 0),  # TN
        ]
        
        metrics = evaluator.calculate_metrics(results)
        
        assert metrics.true_positives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.true_negatives == 1
        assert 0 < metrics.precision < 1
        assert 0 < metrics.recall < 1
    
    def test_calculate_metrics_no_ground_truth(self, evaluator):
        """Test that metrics calculation fails without ground truth"""
        results = [
            RetrievalResult("q1", "c1", 0.9, 1, None),
            RetrievalResult("q1", "c2", 0.1, 0, None),
        ]
        
        with pytest.raises(ValueError):
            evaluator.calculate_metrics(results)


class TestSampleData:
    """Test sample data generation"""
    
    def test_create_sample_data(self):
        """Test that sample data is created correctly"""
        queries, chunks, labels = create_sample_data()
        
        assert len(queries) > 0
        assert len(chunks) == len(queries)
        assert len(labels) == len(queries)
        
        for chunk_list, label_list in zip(chunks, labels):
            assert len(chunk_list) == len(label_list)
            assert all(isinstance(label, int) for label in label_list)
            assert all(label in [0, 1] for label in label_list)


class TestUtils:
    """Test utility functions"""
    
    def test_calculate_metrics_by_query(self):
        """Test per-query metrics calculation"""
        data = {
            'query': ['q1', 'q1', 'q2', 'q2'],
            'is_relevant': [1, 0, 1, 1],
            'ground_truth': [1, 0, 0, 1],
            'relevance_score': [0.9, 0.2, 0.8, 0.95]
        }
        df = pd.DataFrame(data)
        
        query_metrics = calculate_metrics_by_query(df)
        
        assert len(query_metrics) == 2  # Two unique queries
        assert 'precision' in query_metrics.columns
        assert 'recall' in query_metrics.columns
        assert 'f1_score' in query_metrics.columns
    
    def test_analyze_error_patterns(self):
        """Test error pattern analysis"""
        data = {
            'query': ['q1', 'q1', 'q2', 'q2'],
            'retrieved_chunk': ['c1', 'c2', 'c3', 'c4'],
            'is_relevant': [1, 1, 0, 0],
            'ground_truth': [1, 0, 1, 0],
            'relevance_score': [0.9, 0.8, 0.3, 0.1]
        }
        df = pd.DataFrame(data)
        
        analysis = analyze_error_patterns(df)
        
        assert 'total_false_positives' in analysis
        assert 'total_false_negatives' in analysis
        assert analysis['total_false_positives'] == 1
        assert analysis['total_false_negatives'] == 1


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_evaluation(self, tmp_path):
        """Test complete evaluation pipeline"""
        # Create mock evaluator
        with patch('evaluator.Together') as mock_together:
            mock_llm = Mock()
            mock_llm.invoke = Mock(side_effect=["1", "0", "1"] * 5)
            mock_together.return_value = mock_llm
            
            evaluator = RAGRetrievalEvaluator(together_api_key="test_key")
            
            queries, chunks, labels = create_sample_data()
            
            # Run evaluation
            results = evaluator.evaluate_retrieval_batch(
                queries[:2],  # Test with subset
                chunks[:2],
                labels[:2]
            )
            
            assert len(results) > 0
            assert all(isinstance(r, RetrievalResult) for r in results)
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics(results)
            
            assert isinstance(metrics, EvaluationMetrics)
            assert 0 <= metrics.precision <= 1
            assert 0 <= metrics.recall <= 1
            assert 0 <= metrics.accuracy <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])