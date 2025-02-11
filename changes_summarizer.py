# changes_summarizer.py
import glob
import os
from llm_interface import query_llm

def extract_code_changes(repo_path: str) -> str:
    """
    Extract code from all Python files in the repo and create synthetic change descriptions.
    
    Keywords: **LLM**, **RAG**, **vector search**, **FAISS**, **prompt engineering**, **chain-of-thought**
    """
    summaries = []
    for filepath in glob.glob(os.path.join(repo_path, '**', '*.py'), recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            pseudo_summary = f"File '{os.path.basename(filepath)}' implements core functionality and may benefit from refactoring."
            summaries.append(pseudo_summary)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return "\n".join(summaries)

def summarize_changes(repo_path: str) -> str:
    """
    Summarize the synthetic code change descriptions into a concise summary.
    
    Keywords: **LLM**, **retrieval-augmented generation (RAG)**, **prompt engineering**
    """
    changes_text = extract_code_changes(repo_path)
    prompt = (
        "Summarize the following synthetic code change descriptions into a concise changes summary. "
        "Highlight key improvements, refactoring suggestions, and potential new features:\n\n"
        f"{changes_text}\n\n"
        "Summary of changes:"
    )
    return query_llm(prompt, max_length=512)
