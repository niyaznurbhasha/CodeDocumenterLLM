# llm_interface.py
from transformers import pipeline

# Initialize a text-generation pipeline.
# Replace "gpt2" with your locally fine-tuned model if desired.
llm_model = pipeline("text-generation", model="gpt2", device=0)

def query_llm(prompt: str, max_length: int = 512) -> str:
    """
    Query the LLM with a prompt and return generated text.
    
    Keywords: **LLM**, **prompt engineering**, **chain-of-thought**
    """
    response = llm_model(prompt, max_length=max_length, num_return_sequences=1)
    return response[0]['generated_text']
