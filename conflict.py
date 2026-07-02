"""
conflict.py
-----------
Parsing and validation of the LLM's raw text response.
"""

from typing import List, Dict


class ScheduleFormatError(ValueError):
    """Raised when a response is missing the expected 'FINAL SCHEDULE:' marker."""


def extract_final_schedule(draft_response: str) -> List[Dict[str, str]]:
    """Parse the 'FINAL SCHEDULE:' section into a list of session dicts.

    Expects each line in the format: Name | Day | StartTime | EndTime
    Lines that don't match this shape (wrong column count, blank, etc.)
    are skipped rather than raising, since the surrounding reasoning text
    can legitimately contain stray pipe characters.
    """
    if "FINAL SCHEDULE:" not in draft_response:
        raise ScheduleFormatError(
            "Schedule not found in expected format, cannot check conflicts"
        )

    schedule_section = draft_response.split("FINAL SCHEDULE:")[1].strip()

    sessions = []
    for line in schedule_section.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue

        name, day, start, end = parts
        sessions.append({
            "name": name,
            "session_day": day,
            "start_time": start,
            "end_time": end,
        })

    return sessions


def has_conflict(draft_response: str) -> bool:
    """Return True if any two parsed sessions share the same (day, start_time)."""
    sessions = extract_final_schedule(draft_response)

    seen = set()
    for session in sessions:
        key = (session["session_day"], session["start_time"])
        if key in seen:
            print("Initial schedule has conflicts! Proceeding to reflect and fix the schedule")
            return True
        seen.add(key)

    print("Initial schedule is accurate! No reflection required")
    return False
