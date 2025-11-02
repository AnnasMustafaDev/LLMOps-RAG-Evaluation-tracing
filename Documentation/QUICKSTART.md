# 🚀 Quick Start Guide

Get up and running with RAG Evaluation in 5 minutes!

## Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-evaluation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Get API Key (1 minute)

1. Sign up at [Together AI](https://api.together.xyz/)
2. Get your API key from the dashboard
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Add your API key to `.env`:
   ```
   TOGETHER_API_KEY=your_actual_key_here
   ```

## Step 3: Run Your First Evaluation (2 minutes)

### Option A: Use the Demo

```bash
python evaluator.py
```

This will:
- ✅ Evaluate 5 sample queries with 3 chunks each
- ✅ Calculate precision, recall, F1, and accuracy
- ✅ Launch Phoenix UI at http://localhost:6006
- ✅ Save results to CSV files

### Option B: Use Your Own Data

```python
# your_evaluation.py
import os
from evaluator import RAGRetrievalEvaluator

# Initialize
api_key = os.getenv("TOGETHER_API_KEY")
evaluator = RAGRetrievalEvaluator(api_key)

# Your data
queries = ["What is Python?"]
chunks = [
    [
        "Python is a programming language",
        "Snakes are reptiles",
        "Python was created by Guido van Rossum"
    ]
]
labels = [[1, 0, 1]]  # Optional ground truth

# Evaluate
results = evaluator.evaluate_retrieval_batch(queries, chunks, labels)

# Get metrics
metrics = evaluator.calculate_metrics(results)
print(f"Precision: {metrics.precision:.2f}")
print(f"Recall: {metrics.recall:.2f}")
```

## Understanding the Output

### Console Output
```
Evaluating query 1/1: What is Python?...

EVALUATION METRICS
Precision:  1.0000
Recall:     1.0000
Accuracy:   1.0000
F1 Score:   1.0000

Confusion Matrix:
True Positives:  2
False Positives: 0
True Negatives:  1
False Negatives: 0
```

### Files Created
- `retrieval_evaluation_results.csv` - All chunk evaluations
- `evaluation_metrics.csv` - Overall metrics
- Phoenix UI at `http://localhost:6006` - Interactive traces

## Next Steps

### 1. Try the Examples
```bash
python example_usage.py
```
Choose from 6 interactive examples!
### 2. Use Your RAG System

```python
from your_rag_system import retrieve_documents
from evaluator import RAGRetrievalEvaluator

evaluator = RAGRetrievalEvaluator(api_key)

# Get your RAG results
query = "Your question"
rag_results = retrieve_documents(query)

# Evaluate them
results = evaluator.evaluate_retrieval_batch(
    queries=[query],
    retrieved_chunks=[rag_results]
)
```

### 3. Create Ground Truth

Create a CSV file `ground_truth.csv`:
```csv
query,chunk,relevance_label
"What is AI?","AI is artificial intelligence",1
"What is AI?","Pizza is food",0
```

Load and evaluate:
```python
from utils import load_ground_truth_from_csv

queries, chunks, labels = load_ground_truth_from_csv("ground_truth.csv")
results = evaluator.evaluate_retrieval_batch(queries, chunks, labels)
```

### 4. Explore Phoenix

1. Open http://localhost:6006
2. Click on "Traces" to see all LLM calls
3. Analyze patterns in relevant/irrelevant predictions
4. Find systematic errors

## Common Tasks

### Change the Model
```python
evaluator = RAGRetrievalEvaluator(
    api_key,
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
)
```

### Export HTML Report
```python
from utils import export_results_to_html

export_results_to_html(
    results_df,
    metrics.to_dict(),
    "my_report.html"
)
```

### Per-Query Analysis
```python
from utils import calculate_metrics_by_query

query_metrics = calculate_metrics_by_query(results_df)
print(query_metrics)
```

## Troubleshooting

**Problem**: `TOGETHER_API_KEY not set`  
**Solution**: Make sure you created `.env` file with your API key

**Problem**: Phoenix UI not loading  
**Solution**: Check if port 6006 is available or change port:
```bash
export PHOENIX_PORT=6007
```

**Problem**: Rate limit errors  
**Solution**: Add delays between requests:
```python
import time
time.sleep(0.1)  # 100ms delay
```

## What's Next?

- 📖 Read the full [README.md](README.md) for detailed documentation
- 🔬 Check [example_usage.py](example_usage.py) for advanced patterns
- 🧪 Run tests: `pytest test_evaluator.py -v`
- 🎨 Customize the relevance prompt for your domain

## Support

- 🐛 Found a bug? Open an issue
- 💡 Have a question? Check the README
- 🌟 Like the project? Give it a star!

---

**Happy Evaluating! 🎉**

Time from zero to first evaluation: ~5 minutes ⏱️
