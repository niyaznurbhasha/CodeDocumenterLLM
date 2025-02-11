# multi_agent.py
"""
Multi-Agent Orchestration for LLM Tasks

This module demonstrates a simple multi-agent setup:
  - A retrieval agent extracts relevant information.
  - A summarization agent condenses the information using chain-of-thought reasoning.
  - A critique agent refines the summary.
  
Keywords: **multi-agent**, **chain-of-thought (CoT)**, **prompt engineering**, **LLM**
"""
from llm_interface import query_llm

def retrieval_agent(prompt: str) -> str:
    """
    Simulate a retrieval agent to fetch relevant content.
    """
    retrieval_prompt = f"Retrieve relevant details for: {prompt}"
    return query_llm(retrieval_prompt, max_length=256)

def summarization_agent(text: str) -> str:
    """
    Summarize the retrieved content using chain-of-thought reasoning.
    """
    summary_prompt = f"Summarize the following content in a detailed, step-by-step manner:\n\n{text}\n\nSummary:"
    return query_llm(summary_prompt, max_length=256)

def critique_agent(summary: str) -> str:
    """
    Critique and refine the summary to add clarity and detail.
    """
    critique_prompt = f"Improve the following summary by adding further details and clarifications:\n\n{summary}\n\nImproved summary:"
    return query_llm(critique_prompt, max_length=256)

def multi_agent_workflow(initial_prompt: str) -> str:
    """
    Orchestrate a multi-agent workflow:
      1. Retrieval agent gathers data.
      2. Summarization agent condenses the data.
      3. Critique agent refines the summary.
      
    Keywords: **multi-agent**, **chain-of-thought**, **prompt engineering**
    """
    retrieved = retrieval_agent(initial_prompt)
    summarized = summarization_agent(retrieved)
    critiqued = critique_agent(summarized)
    return critiqued

if __name__ == "__main__":
    test_prompt = "Describe the core functionalities of this code repository and potential refactoring improvements."
    result = multi_agent_workflow(test_prompt)
    print("Multi-Agent Final Output:")
    print(result)
