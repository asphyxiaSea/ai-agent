import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:8001")
PADDLE_EXTRACT_PATH_URL = os.environ.get(
    "PADDLE_EXTRACT_PATH_URL",
    "http://localhost:8003/paddle/pp-structurev3/predict_path",
)

DEFAULT_MODEL_NAME = "llama3.1:8b"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000
