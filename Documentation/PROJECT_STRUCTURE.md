# Project Structure

Complete file structure for the RAG Evaluation System.

```
rag-evaluation/
│
├── 📄 Core Application Files
│   ├── evaluator.py              # Main evaluation logic and RAGRetrievalEvaluator class
│   ├── config.py                 # Configuration management (LLM, Phoenix, Evaluation)
│   └── utils.py                  # Utility functions (metrics, I/O, analysis)
│
├── 📚 Documentation
│   ├── README.md                 # Complete project documentation
│   ├── QUICKSTART.md             # 5-minute getting started guide
│   ├── PROJECT_STRUCTURE.md      # This file
│   └── LICENSE                   # MIT License (create if needed)
│
├── 🧪 Testing & Quality
│   ├── test_evaluator.py         # Comprehensive unit tests
│   ├── .github/
│   │   └── workflows/
│   │       └── ci.yml            # GitHub Actions CI/CD pipeline
│   └── .flake8                   # Linter configuration (optional)
│
├── 🐳 Docker & Deployment
│   ├── Dockerfile                # Docker image definition
│   ├── docker-compose.yml        # Multi-container setup
│   └── .dockerignore             # Docker ignore patterns (create if needed)
│
├── 📦 Package Management
│   ├── requirements.txt          # Python dependencies
│   ├── setup.py                  # Package installation script
│   ├── Makefile                  # Convenient command shortcuts
│   └── pyproject.toml            # Modern Python packaging (optional)
│
├── ⚙️ Configuration
│   ├── .env                      # Environment variables (create from .env.example)
│   ├── .env.example              # Environment template
│   └── .gitignore                # Git ignore patterns
│
├── 📊 Example & Usage
│   ├── example_usage.py          # 6 detailed usage examples
│   └── notebooks/                # Jupyter notebooks (optional)
│       ├── tutorial.ipynb
│       └── advanced_usage.ipynb
│
├── 📁 Data Directories
│   ├── data/                     # Input data (create manually)
│   │   ├── queries.txt
│   │   ├── ground_truth.csv
│   │   └── custom_dataset/
│   │
│   └── results/                  # Output files (auto-created)
│       ├── retrieval_evaluation_results.csv
│       ├── evaluation_metrics.csv
│       ├── evaluation_report.html
│       └── *.json
│
└── 🔧 Development (Optional)
    ├── .vscode/                  # VS Code settings
    │   └── settings.json
    ├── .pre-commit-config.yaml   # Pre-commit hooks
    └── tox.ini                   # Multi-environment testing
```

## File Descriptions

### Core Application Files

#### `evaluator.py` (Main Module)
- **Classes:**
  - `RetrievalResult`: Data class for storing evaluation results
  - `EvaluationMetrics`: Data class for metrics (precision, recall, etc.)
  - `RAGRetrievalEvaluator`: Main evaluator using LangChain + Together AI
  - `PhoenixRAGTracer`: Phoenix integration for tracing
- **Functions:**
  - `create_sample_data()`: Generate sample data for testing
  - `main()`: Entry point for CLI execution

#### `config.py` (Configuration)
- **Classes:**
  - `LLMConfig`: LLM provider and model settings
  - `EvaluationConfig`: Evaluation parameters
  - `PhoenixConfig`: Phoenix tracing settings
  - `RAGEvalConfig`: Main configuration combining all settings
- **Methods:**
  - `from_defaults()`: Load default configuration
  - `from_env()`: Load configuration from environment variables

#### `utils.py` (Utilities)
- **Functions:**
  - `load_queries_from_file()`: Load queries from text file
  - `load_ground_truth_from_csv()`: Load labeled data from CSV
  - `save_results_to_json()`: Export results to JSON
  - `calculate_metrics_by_query()`: Per-query metric analysis
  - `analyze_error_patterns()`: Error analysis (FP/FN patterns)
  - `export_results_to_html()`: Generate HTML reports
  - `print_metrics_summary()`: Pretty-print metrics

### Testing & Quality

#### `test_evaluator.py` (Tests)
- Unit tests for all major components
- Integration tests for end-to-end workflow
- Mock LLM for deterministic testing
- Coverage: ~90%+

#### `.github/workflows/ci.yml` (CI/CD)
- Automated testing on push/PR
- Multi-OS testing (Ubuntu, macOS, Windows)
- Multi-Python version (3.8, 3.9, 3.10, 3.11)
- Code coverage reporting to Codecov
- Security scanning with Bandit

### Docker & Deployment

#### `Dockerfile`
- Python 3.10 slim base image
- Installs all dependencies
- Exposes Phoenix port 6006
- Health check included

#### `docker-compose.yml`
- Main evaluation service
- Optional Jupyter notebook service
- Volume mounts for data and results
- Environment variable configuration

### Package Management

#### `requirements.txt`
Core dependencies:
- `langchain`, `langchain-together`, `together`
- `arize-phoenix`
- `pandas`, `numpy`, `scikit-learn`
- `chromadb`, `sentence-transformers` (optional)

#### `setup.py`
- Package metadata and installation
- Entry points for CLI commands:
  - `rag-eval`: Run main evaluation
  - `rag-eval-examples`: Run examples
- Development dependencies specification

#### `Makefile`
Convenient commands:
- `make setup`: Complete setup
- `make test`: Run tests
- `make run`: Run evaluation
- `make clean`: Clean temporary files
- `make format`: Format code

### Example & Usage

#### `example_usage.py`
Six comprehensive examples:
1. Basic evaluation
2. Phoenix tracing integration
3. Error analysis
4. Custom configuration
5. Export reports (HTML, JSON, CSV)
6. Batch processing

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | ≥0.1.0 | LLM orchestration |
| langchain-together | ≥0.0.1 | Together AI integration |
| arize-phoenix | ≥4.0.0 | Observability and tracing |
| pandas | ≥2.0.0 | Data manipulation |
| scikit-learn | ≥1.3.0 | Metrics calculation |
| together | ≥1.0.0 | Together AI client |

## Data Flow

```
1. Input Data
   └── queries.txt, ground_truth.csv, or Python lists

2. RAGRetrievalEvaluator
   ├── LangChain + Together AI for relevance scoring
   └── Binary classification (0 or 1) per chunk

3. Metrics Calculation
   ├── Precision, Recall, Accuracy, F1
   └── Confusion matrix

4. Output
   ├── CSV files (results + metrics)
   ├── HTML report
   ├── JSON export
   └── Phoenix UI traces
```

## Setup Instructions

### Basic Setup
```bash
# 1. Clone and navigate
git clone <repo-url>
cd rag-evaluation

# 2. Install
make setup

# 3. Configure
cp .env.example .env
# Edit .env with your API key

# 4. Run
make run
```

### Docker Setup
```bash
# 1. Build
docker-compose build

# 2. Configure
cp .env.example .env
# Edit .env with your API key

# 3. Run
docker-compose up
```

## Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
vim evaluator.py

# 3. Format code
make format

# 4. Run tests
make test

# 5. Check everything
make check  # Runs lint, type-check, and test

# 6. Commit and push
git commit -am "Add feature"
git push origin feature/my-feature
```

## Common Customizations

### Add New Metric
Edit `utils.py`:
```python
def calculate_custom_metric(results):
    # Your metric calculation
    pass
```

### Change LLM Provider
Edit `config.py` or `.env`:
```python
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
```

### Custom Relevance Prompt
Edit `evaluator.py`:
```python
self.relevance_prompt = PromptTemplate(
    template="Your custom prompt",
    input_variables=["query", "document"]
)
```

## Output Formats

### CSV Format
```csv
query,retrieved_chunk,relevance_score,is_relevant,ground_truth
"What is AI?","AI is artificial intelligence",0.95,1,1
```

### JSON Format
```json
{
  "query": "What is AI?",
  "retrieved_chunk": "AI is artificial intelligence",
  "relevance_score": 0.95,
  "is_relevant": 1,
  "ground_truth": 1
}
```

### HTML Report
Interactive HTML with:
- Metrics dashboard
- Detailed results table
- Sortable columns
- Responsive design

## Version Control

### `.gitignore` (Recommended)
```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.env

# Results
results/*.csv
results/*.json
results/*.html

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

## Additional Resources

- **Phoenix Docs**: https://docs.arize.com/phoenix
- **LangChain Docs**: https://python.langchain.com
- **Together AI Docs**: https://docs.together.ai
- **scikit-learn Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html

---

**Need help?** Open an issue or check the QUICKSTART.md guide!