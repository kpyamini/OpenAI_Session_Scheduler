"""
scheduling_chain.py
--------------------
Wires together the actual LangChain pipeline:

    prompt -> chat -> raw text -> log reasoning -> reflect if needed -> structured output

Why this exists: this is the only file that needs to know LangChain's
Runnable/pipe syntax. Everything else (prompts, parsing, models, data) is
plain Python that those Runnables consume. If LangChain's API changes, or
you swap frameworks, this is the file that absorbs the impact.
"""

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from config import get_chat_model
from models import Schedule
from prompts import scheduling_prompt, REFLECTION_TEMPLATE
from conflict import has_conflict
from utils import print_reasoning

chat = get_chat_model()


def reflect_if_needed(draft_response: str) -> str:
    """Return the draft unchanged if no conflicts are found.

    If a conflict is found, sends the draft back to the LLM with a
    correction prompt and returns the corrected text instead.
    """
    if not has_conflict(draft_response):
        return draft_response

    reflection_prompt = REFLECTION_TEMPLATE.format(draft_schedule=draft_response)
    reflection_response = chat.invoke(reflection_prompt)
    return reflection_response.content


def build_chain():
    """Assemble and return the full scheduling pipeline as a single Runnable.

    Stages:
      1. initial_chain: prompt -> chat -> string -> log the reasoning trace.
      2. reflect_if_needed: only re-invokes the LLM if a conflict is detected,
         so the happy path costs exactly one LLM call.
      3. structured_chat: coerces the final text into a validated `Schedule`.
    """
    initial_chain = (
        scheduling_prompt
        | chat
        | StrOutputParser()
        | RunnableLambda(print_reasoning)
    )

    structured_chat = chat.with_structured_output(Schedule)

    return initial_chain | RunnableLambda(reflect_if_needed) | structured_chat
