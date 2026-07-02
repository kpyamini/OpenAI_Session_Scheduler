"""
models.py
---------
Pydantic schemas describing a single scheduled session and the overall
weekly schedule.

These serve two purposes:
1. They constrain the LLM's structured output (via `with_structured_output`).
2. They give every other module in the pipeline a single, type-checked
   definition of "what a session looks like" to import, instead of passing
   around loosely-shaped dicts.
"""

from typing import List
from pydantic import BaseModel, Field


class Session(BaseModel):
    name: str = Field(description="Name of Client")
    session_day: str = Field(description="Day of the scheduled session")
    start_time: str = Field(description="Start time of the scheduled session")
    end_time: str = Field(description="End time of the scheduled session")


class Schedule(BaseModel):
    sessions: List[Session]
