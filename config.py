"""
Configuration module for RAG Evaluation System
"""

from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: str = "together"
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    temperature: float = 0.0
    max_tokens: int = 512
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if self.api_key is None:
            if self.provider == "together":
                self.api_key = os.getenv("TOGETHER_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key not found for provider: {self.provider}")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings"""
    relevance_threshold: float = 0.7
    batch_size: int = 10
    enable_phoenix: bool = True
    save_results: bool = True
    results_dir: str = "results"
    
    # Metric calculation settings
    zero_division: int = 0
    average: str = "binary"
    
    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)


@dataclass
class PhoenixConfig:
    """Configuration for Phoenix tracing"""
    enable_tracing: bool = True
    project_name: str = "rag_evaluation"
    host: str = "0.0.0.0"
    port: int = 6006


@dataclass
class RAGEvalConfig:
    """Main configuration combining all settings"""
    llm: LLMConfig
    evaluation: EvaluationConfig
    phoenix: PhoenixConfig
    
    @classmethod
    def from_defaults(cls):
        """Create configuration with default values"""
        return cls(
            llm=LLMConfig(),
            evaluation=EvaluationConfig(),
            phoenix=PhoenixConfig()
        )
    
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        return cls(
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "together"),
                model_name=os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0"))
            ),
            evaluation=EvaluationConfig(
                relevance_threshold=float(os.getenv("RELEVANCE_THRESHOLD", "0.7")),
                batch_size=int(os.getenv("BATCH_SIZE", "10")),
                results_dir=os.getenv("RESULTS_DIR", "results")
            ),
            phoenix=PhoenixConfig(
                enable_tracing=os.getenv("ENABLE_PHOENIX", "true").lower() == "true",
                project_name=os.getenv("PHOENIX_PROJECT", "rag_evaluation"),
                port=int(os.getenv("PHOENIX_PORT", "6006"))
            )
        )