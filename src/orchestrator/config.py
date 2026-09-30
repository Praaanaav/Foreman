import os

from dotenv import load_dotenv

load_dotenv()

SUPERVISOR_MODEL = os.getenv(
    "SUPERVISOR_MODEL",
    "groq/llama-3.3-70b-versatile"
)

WORKER_MODEL = os.getenv(
    "WORKER_MODEL",
    "groq/llama-3.3-70b-versatile"
)