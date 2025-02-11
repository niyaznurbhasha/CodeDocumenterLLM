# inference.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI(
    title="LLM Code Documentation & Refactoring API",
    description="An API that leverages an LLM for code refactoring, documentation generation, and synthetic change summarization using RAG and multi-agent workflows.",
)

# Load your fine-tuned model (adjust model path as necessary)
MODEL_PATH = "./fine_tuned_codet5"
llm_model = pipeline("text-generation", model=MODEL_PATH, device=0)

class CodeRequest(BaseModel):
    code: str

class RefactorResponse(BaseModel):
    refactored_code: str

@app.post("/refactor", response_model=RefactorResponse)
def refactor_code_endpoint(request: CodeRequest):
    """
    Endpoint to refactor code and generate detailed documentation.
    
    Keywords: **LLM**, **code refactoring**, **docstrings**, **prompt engineering**
    """
    prompt = (
        "Refactor the following code to an object-oriented design with detailed inline comments and docstrings:\n\n"
        f"{request.code}\n\n"
        "Refactored code:"
    )
    try:
        response = llm_model(prompt, max_length=1024, num_return_sequences=1)
        generated_text = response[0]['generated_text']
        return RefactorResponse(refactored_code=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "LLM Code Documentation & Refactoring API is up and running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
