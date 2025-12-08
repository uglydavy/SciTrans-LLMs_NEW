"""Setup script for SciTrans-LLMs."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
readme = Path("README.md").read_text(encoding="utf-8")

setup(
    name="scitrans-llms",
    version="2.0.0",
    description="Advanced Scientific Document Translation with LLMs",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@university.edu",
    url="https://github.com/yourusername/SciTrans-LLMs_NEW",
    license="MIT",
    
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    
    python_requires=">=3.9",
    
    install_requires=[
        "numpy>=1.24.0,<2.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "PyMuPDF>=1.23.0",
        "pypdf>=3.0.0",
        "pdfplumber>=0.9.0",
        "Pillow>=10.0.0",
        "openai>=1.0.0",
        "anthropic>=0.25.0",
        "deep-translator>=1.11.0",
        "ollama>=0.1.0",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "click>=8.1.0",
        "typer>=0.9.0",
        "loguru>=0.7.0",
        "diskcache>=5.6.0",
        "httpx>=0.24.0"
    ],
    
    extras_require={
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "scikit-learn>=1.3.0"
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0"
        ],
        "gui": [
            "gradio>=4.0.0",
            "plotly>=5.0.0"
        ]
    },
    
    entry_points={
        "console_scripts": [
            "scitrans=cli.commands.main:cli",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    keywords="translation nlp scientific llm pdf latex",
    
    project_urls={
        "Bug Reports": "https://github.com/yourusername/SciTrans-LLMs_NEW/issues",
        "Source": "https://github.com/yourusername/SciTrans-LLMs_NEW",
    },
)
