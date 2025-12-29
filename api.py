import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from dotenv import load_dotenv

# --- Tracing Setup ---
# This block configures the tracer to send data to a local tracing instance.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use host.docker.internal for Docker container to access host services
tracing_host = os.getenv("PHOENIX_HOST", "host.docker.internal")
tracing_endpoint = f"http://{tracing_host}:6006"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = tracing_endpoint

try:
    from phoenix.otel import register
    tracing_provider = register(
        project_name="default",
        endpoint=f"{tracing_endpoint}/v1/traces",
        auto_instrument=True
    )
    logging.info(f"✅ Tracing successfully initialized for API server at {tracing_endpoint}")
except ImportError as error:
    logging.warning(f"⚠️  Tracing module not found: {error}")
except Exception as error:
    logging.warning(f"⚠️  Could not initialize tracing: {error}")
# --- End of Tracing Setup ---

# Ensure the project root is in the Python path
import sys
application_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, application_root)

# Import the crew creation function
from src.rag_system.crew import create_rag_crew

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI(
    title="Application RAG API",
    description="An API server for the agentic RAG pipeline.",
    version="1.0.0",
)

# Add CORS middleware to allow UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request model
class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]

@app.get("/v1/models")
def list_models():
    """
    OpenAI-compatible endpoint to list available models.
    Required for UI model discovery.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "app-rag-model",
                "object": "model",
                "created": 1677652288,
                "owned_by": "app-rag-model",
                "permission": [],
                "root": "app-rag-model",
                "parent": None,
                "max_tokens": 131072,
                "context_length": 131072
            }
        ]
    }

@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible endpoint to interact with the RAG pipeline.
    """
    # Extract the last user message as the query
    user_message = next(
        (msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"),
        None
    )

    if not user_message:
        return {"error": "No user message found"}

    print(f"Received query for API: {user_message}")

    # Execute the application crew with the user's query
    application_crew = create_rag_crew(user_message)
    execution_result = application_crew.kickoff()

    # Format the response to be compatible with OpenAI API format
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": str(execution_result),
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)