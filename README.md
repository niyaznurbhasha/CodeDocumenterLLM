# CodeDocumenterLLM

**CodeDocumenterLLM** is an AI-powered tool that automatically documents, summarizes, and refactors code using Large Language Models (LLMs). It helps improve code maintainability and collaboration by generating clear, up-to-date documentation directly from your codebase.

## Features

- 🔍 **Code Change Summarization** — Summarizes recent changes in the codebase.
- 🧠 **Automated README Generation** — Creates README files based on current code structure.
- ✨ **Code Refactoring Assistance** — Suggests and applies code improvements.
- 🔎 **Vector-Based Code Search** — Enables semantic search of code using embeddings.
- ⚙️ **Quantization Support** — Reduces model size for efficient inference.
- 🤖 **Multi-Agent Collaboration** — Coordinates multiple AI agents to handle complex documentation workflows.

## File Overview

- `changes_summarizer.py` — Summarizes recent edits in your codebase.
- `code_refactor.py` — Refactors code for readability and performance.
- `inference.py` — Handles model inference.
- `llm_interface.py` — Communicates with the underlying LLMs.
- `multi_agent.py` — Manages collaboration between multiple LLM agents.
- `quantization.py` — Optimizes models using quantization.
- `readme_generator.py` — Automatically generates README files from code.
- `train.py` — Trains the models used by the tool.
- `vector_search.py` — Enables semantic search using vector embeddings.
- `requirements.txt` — Lists the Python packages needed.

## Getting Started

### Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
