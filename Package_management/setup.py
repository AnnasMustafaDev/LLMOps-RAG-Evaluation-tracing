"""
Setup script for RAG Evaluation System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rag-evaluation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="RAG Retrieval and Generation Evaluation System with Arize Phoenix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-evaluation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-eval=evaluator:main",
            "rag-eval-examples=example_usage:run_all_examples",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml"],
    },
)