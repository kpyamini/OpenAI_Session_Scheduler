"""
config.py
---------
Centralizes environment loading and LLM client construction.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# --- LLM settings -----------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 2000
TIMEOUT_SECONDS = 60

# --- File paths --------------------------------------------------------------
AVAILABILITY_CSV_PATH = "availability.csv"
OUTPUT_CSV_PATH = "scheduled_sessions.csv"


def get_chat_model() -> ChatOpenAI:
    """Construct a configured ChatOpenAI client.
    """
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_SECONDS,
    )
