import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

SUPERVISOR_MODEL = os.getenv(
    "SUPERVISOR_MODEL",
    "groq/llama-3.3-70b-versatile",
)

WORKER_MODEL = os.getenv(
    "WORKER_MODEL",
    "groq/llama-3.3-70b-versatile",
)

# Where traveler memory and outputs live (overridable for tests)
DATA_DIR = Path(os.getenv("FOREMAN_DATA_DIR", str(BASE_DIR / "data")))
OUTPUT_DIR = Path(os.getenv("FOREMAN_OUTPUT_DIR", str(BASE_DIR / "outputs")))

# Agent behaviour limits
MAX_TOOL_ITERS = 4      # tool-loop iterations per specialist run
MAX_REPLANS = 2         # reviewer-reject replan cycles before escalation
LLM_RETRIES = 2         # retries on LLM errors inside a node
DEFAULT_TRAVELER = "guest"
