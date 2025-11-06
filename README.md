# RAG Retrieval & Generation Evaluation System

A comprehensive evaluation framework for Retrieval-Augmented Generation (RAG) systems using Arize Phoenix for observability and LangChain with Together AI for LLM-based relevance scoring.

## Features

- 🎯 **Binary Relevance Evaluation**: Each retrieved chunk is scored as 0 (not relevant) or 1 (relevant) using an LLM
- 📊 **Comprehensive Metrics**: Precision, Recall, Accuracy, F1-Score, and confusion matrix analysis
- 🔍 **Phoenix Integration**: Real-time tracing and observability with Arize Phoenix
- 🤖 **LangChain + Together AI**: Leverages Together AI's fast inference with LangChain
- 📈 **Detailed Reports**: HTML, CSV, and JSON output formats
- ✅ **Full Test Coverage**: Comprehensive unit and integration tests
- ✅ **Docker Container**: Docker File along with yml
- ✅ **CI/CD pipeline**: CI/CD pipeline integrated using github workflows

## Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Retrieval  │ ← Your retrieval system
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  LLM-Based Relevance Eval   │ ← This system
│  (Together AI + LangChain)  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Metrics Calculation        │
│  (Precision, Recall, F1)    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Phoenix Tracing & Reports  │
└─────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Together AI API key

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag-evaluation
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
cat > .env << EOF
TOGETHER_API_KEY=your_together_api_key_here
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
RELEVANCE_THRESHOLD=0.7
ENABLE_PHOENIX=true
EOF
```

## Project Structure

```
rag-evaluation/
├── evaluator.py           # Main evaluation logic
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── test_evaluator.py      # Unit tests
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── README.md              # This file
├── example_usage.py       # Example usage script
└── results/               # Output directory (auto-created)
    ├── retrieval_evaluation_results.csv
    ├── evaluation_metrics.csv
    └── evaluation_report.html
```

## Usage

### Basic Usage

```python
import os
from evaluator import RAGRetrievalEvaluator, PhoenixRAGTracer, create_sample_data

# Initialize evaluator
together_api_key = os.getenv("TOGETHER_API_KEY")
evaluator = RAGRetrievalEvaluator(together_api_key)

# Initialize Phoenix tracer
phoenix_tracer = PhoenixRAGTracer()

# Prepare your data
queries = ["What is machine learning?"]
retrieved_chunks = [
    [
        "Machine learning is a subset of AI...",
        "Pizza is a popular food...",
        "ML algorithms learn from data..."
    ]
]
ground_truth_labels = [[1, 0, 1]]  # Optional

# Run evaluation
results = evaluator.evaluate_retrieval_batch(
    queries=queries,
    retrieved_chunks=retrieved_chunks,
    ground_truth_labels=ground_truth_labels
)

# Calculate metrics
metrics = evaluator.calculate_metrics(results)

print(f"Precision: {metrics.precision:.4f}")
print(f"Recall: {metrics.recall:.4f}")
print(f"F1 Score: {metrics.f1_score:.4f}")

# Log to Phoenix
results_df = phoenix_tracer.log_retrieval_results(results)
```

### Running the Demo

```bash
# Run the main evaluation script
python evaluator.py
```

This will:
1. Start Phoenix UI (available at http://localhost:6006)
2. Run evaluation on sample data
3. Generate metrics and reports
4. Save results to CSV files

### Custom Data Format

**CSV Format** (for `utils.load_ground_truth_from_csv`):
```csv
query,chunk,relevance_label
"What is ML?","Machine learning is...",1
"What is ML?","Pizza is food...",0
```

**Python Format**:
```python
queries = ["query1", "query2"]
retrieved_chunks = [
    ["chunk1_for_q1", "chunk2_for_q1"],
    ["chunk1_for_q2", "chunk2_for_q2"]
]
ground_truth_labels = [
    [1, 0],  # Labels for query1 chunks
    [1, 1]   # Labels for query2 chunks
]
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOGETHER_API_KEY` | Together AI API key | Required |
| `LLM_MODEL` | Model name | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |
| `LLM_TEMPERATURE` | LLM temperature | `0.0` |
| `RELEVANCE_THRESHOLD` | Relevance threshold | `0.7` |
| `ENABLE_PHOENIX` | Enable Phoenix tracing | `true` |
| `PHOENIX_PORT` | Phoenix UI port | `6006` |
| `RESULTS_DIR` | Output directory | `results` |

### Programmatic Configuration

```python
from config import RAGEvalConfig, LLMConfig, EvaluationConfig

config = RAGEvalConfig(
    llm=LLMConfig(
        provider="together",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        temperature=0.0
    ),
    evaluation=EvaluationConfig(
        relevance_threshold=0.8,
        batch_size=20
    )
)
```

## Metrics Explained

### Binary Classification Metrics

- **Precision**: Of all chunks marked as relevant, what percentage were actually relevant?
  ```
  Precision = TP / (TP + FP)
  ```

- **Recall**: Of all actually relevant chunks, what percentage did we identify?
  ```
  Recall = TP / (TP + FN)
  ```

- **Accuracy**: Overall correctness of predictions
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```

- **F1 Score**: Harmonic mean of precision and recall
  ```
  F1 = 2 * (Precision * Recall) / (Precision + Recall)
  ```

### Confusion Matrix

```
                    Predicted
                    Relevant    Not Relevant
Actual  Relevant    TP          FN
        Not Relevant FP          TN
```

## Phoenix Observability

Phoenix provides real-time observability for your RAG evaluation:

1. **Start Phoenix**: Automatically starts when running the evaluator
2. **Access UI**: Navigate to http://localhost:6006
3. **View Traces**: See all LLM calls, latencies, and relevance decisions
4. **Analyze Patterns**: Identify systematic errors and biases

## Testing

```bash
# Run all tests
pytest test_evaluator.py -v

# Run specific test class
pytest test_evaluator.py::TestRAGRetrievalEvaluator -v

# Run with coverage
pytest test_evaluator.py --cov=evaluator --cov-report=html
```

## Advanced Usage

### Custom Relevance Prompt

```python
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    template="""Evaluate if this document answers the query.
Query: {query}
Document: {document}

Answer only 1 (relevant) or 0 (not relevant):""",
    input_variables=["query", "document"]
)

evaluator.relevance_prompt = custom_prompt
evaluator.chain = custom_prompt | evaluator.llm
```

### Batch Processing with Progress

```python
from tqdm import tqdm

results = []
for query, chunks in tqdm(zip(queries, retrieved_chunks), total=len(queries)):
    batch_results = evaluator.evaluate_retrieval_batch(
        queries=[query],
        retrieved_chunks=[chunks]
    )
    results.extend(batch_results)
```

### Per-Query Analysis

```python
from utils import calculate_metrics_by_query

query_metrics = calculate_metrics_by_query(results_df)
print(query_metrics)
```

### Error Pattern Analysis

```python
from utils import analyze_error_patterns

error_analysis = analyze_error_patterns(results_df)
print(f"False Positives: {error_analysis['total_false_positives']}")
print(f"False Negatives: {error_analysis['total_false_negatives']}")
```

## Output Files

### `retrieval_evaluation_results.csv`
Contains all evaluation results with columns:
- `query`: The user query
- `retrieved_chunk`: The retrieved document chunk
- `relevance_score`: Confidence score (0.0-1.0)
- `is_relevant`: Binary prediction (0 or 1)
- `ground_truth`: Actual label (if provided)

### `evaluation_metrics.csv`
Contains overall metrics:
- Precision, Recall, Accuracy, F1-Score
- True/False Positives/Negatives

### `evaluation_report.html`
Interactive HTML report with tables and metrics

## Troubleshooting

### Common Issues

**1. Phoenix not starting**
```bash
# Check if port 6006 is available
lsof -i :6006

# Use different port
export PHOENIX_PORT=6007
```

**2. Together AI rate limits**
```python
# Add delay between requests
import time
time.sleep(0.1)  # 100ms delay
```

**3. Out of memory with large batches**
```python
# Reduce batch size
config.evaluation.batch_size = 5
```

## Best Practices

1. **Start with small samples**: Test with 5-10 queries first
2. **Use temperature=0**: For consistent relevance evaluation
3. **Validate ground truth**: Ensure labels are accurate
4. **Monitor Phoenix**: Check for systematic biases
5. **Iterate on prompts**: Customize relevance prompt for your domain
6. **Track metrics over time**: Store results for trend analysis

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - Observability platform
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [Together AI](https://www.together.ai/) - Fast LLM inference

## Support

For issues and questions:
- Open a GitHub issue
- Check the [documentation](https://docs.arize.com/phoenix)
- Review example scripts in the repository

## Roadmap

- [ ] Support for additional LLM providers (OpenAI, Anthropic)
- [ ] Advanced Phoenix evaluations (hallucination, toxicity)
- [ ] Multi-metric optimization
- [ ] A/B testing framework
- [ ] Real-time streaming evaluation
- [ ] Vector store integration

---

**Happy Evaluating! 🚀**
