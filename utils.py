"""
Utility functions for RAG evaluation
"""

import json
import csv
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def load_queries_from_file(filepath: str) -> List[str]:
    """
    Load queries from a text file (one query per line)
    
    Args:
        filepath: Path to the file
        
    Returns:
        List of queries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(queries)} queries from {filepath}")
    return queries


def load_ground_truth_from_csv(filepath: str) -> Tuple[List[str], List[List[str]], List[List[int]]]:
    """
    Load ground truth data from CSV
    Expected format: query, chunk, relevance_label
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (queries, chunks, labels)
    """
    df = pd.read_csv(filepath)
    
    # Group by query
    grouped_data = {}
    for _, row in df.iterrows():
        query = row['query']
        if query not in grouped_data:
            grouped_data[query] = {'chunks': [], 'labels': []}
        grouped_data[query]['chunks'].append(row['chunk'])
        grouped_data[query]['labels'].append(int(row['relevance_label']))
    
    queries = list(grouped_data.keys())
    chunks = [grouped_data[q]['chunks'] for q in queries]
    labels = [grouped_data[q]['labels'] for q in queries]
    
    logger.info(f"Loaded {len(queries)} queries with ground truth from {filepath}")
    return queries, chunks, labels


def save_results_to_json(results: List[Dict[str, Any]], filepath: str):
    """
    Save evaluation results to JSON
    
    Args:
        results: List of result dictionaries
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {filepath}")


def save_metrics_report(metrics: Dict[str, Any], filepath: str):
    """
    Save metrics report to JSON with timestamp
    
    Args:
        metrics: Dictionary of metrics
        filepath: Output file path
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Metrics report saved to {filepath}")


def calculate_metrics_by_query(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-query metrics
    
    Args:
        df: DataFrame with query, is_relevant, and ground_truth columns
        
    Returns:
        DataFrame with per-query metrics
    """
    query_metrics = []
    
    for query in df['query'].unique():
        query_df = df[df['query'] == query]
        
        if 'ground_truth' in query_df.columns and query_df['ground_truth'].notna().any():
            y_true = query_df['ground_truth'].values
            y_pred = query_df['is_relevant'].values
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
            
            query_metrics.append({
                'query': query,
                'num_chunks': len(query_df),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            })
    
    return pd.DataFrame(query_metrics)


def print_metrics_summary(metrics_dict: Dict[str, float], title: str = "Metrics Summary"):
    """
    Pretty print metrics summary
    
    Args:
        metrics_dict: Dictionary of metric names and values
        title: Title for the summary
    """
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)
    
    for metric_name, value in metrics_dict.items():
        formatted_name = metric_name.replace('_', ' ').title()
        if isinstance(value, float):
            print(f"{formatted_name:.<40} {value:.4f}")
        else:
            print(f"{formatted_name:.<40} {value}")
    
    print("="*60 + "\n")


def create_confusion_matrix_report(tp: int, fp: int, fn: int, tn: int) -> str:
    """
    Create a formatted confusion matrix report
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        tn: True negatives
        
    Returns:
        Formatted string report
    """
    total = tp + fp + fn + tn
    
    report = f"""
Confusion Matrix:
                    Predicted
                    Relevant    Not Relevant
Actual  Relevant    {tp:<12} {fn:<12}
        Not Relevant{fp:<12} {tn:<12}

Total Predictions: {total}
Relevant Predicted: {tp + fp} ({(tp + fp)/total*100:.1f}%)
Relevant Actual: {tp + fn} ({(tp + fn)/total*100:.1f}%)
"""
    return report


def export_results_to_html(df: pd.DataFrame, metrics: Dict[str, Any], output_path: str):
    """
    Export results to an HTML report
    
    Args:
        df: Results DataFrame
        metrics: Metrics dictionary
        output_path: Output HTML file path
    """
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metrics {{ background-color: #e7f3fe; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            .metric-item {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>RAG Retrieval Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="metrics">
            <h2>Overall Metrics</h2>
            <div class="metric-item"><strong>Precision:</strong> {metrics.get('precision', 0):.4f}</div>
            <div class="metric-item"><strong>Recall:</strong> {metrics.get('recall', 0):.4f}</div>
            <div class="metric-item"><strong>Accuracy:</strong> {metrics.get('accuracy', 0):.4f}</div>
            <div class="metric-item"><strong>F1 Score:</strong> {metrics.get('f1_score', 0):.4f}</div>
        </div>
        
        <h2>Detailed Results</h2>
        {df.to_html(index=False, classes='results-table')}
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    logger.info(f"HTML report saved to {output_path}")


def analyze_error_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze patterns in false positives and false negatives
    
    Args:
        df: Results DataFrame with ground_truth and predictions
        
    Returns:
        Dictionary of error analysis
    """
    if 'ground_truth' not in df.columns:
        return {}
    
    df_labeled = df[df['ground_truth'].notna()].copy()
    
    false_positives = df_labeled[(df_labeled['ground_truth'] == 0) & (df_labeled['is_relevant'] == 1)]
    false_negatives = df_labeled[(df_labeled['ground_truth'] == 1) & (df_labeled['is_relevant'] == 0)]
    
    analysis = {
        'total_false_positives': len(false_positives),
        'total_false_negatives': len(false_negatives),
        'fp_avg_confidence': false_positives['relevance_score'].mean() if len(false_positives) > 0 else 0,
        'fn_avg_confidence': false_negatives['relevance_score'].mean() if len(false_negatives) > 0 else 0,
        'fp_queries': false_positives['query'].unique().tolist()[:5],  # Top 5
        'fn_queries': false_negatives['query'].unique().tolist()[:5],  # Top 5
    }
    
    return analysis