"""
prompts.py
----------
All prompt text lives here, separated from chain-wiring and parsing logic.
"""

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

SCHEDULING_TEMPLATE = """
Act as an assistant to schedule appointments for clients for upcoming week. Starting from Monday to Saturday.
Follow below rules while scheduling:
1. Session duration: 1 hour.
2. Schedule only from trainer's available time slots.
3. Below is the full list of available 1-hour slots for the week.
```{available_slots}```
Assign clients to slots from this list. As you assign each session, remove that slot from
the available pool so it cannot be assigned again. Process clients one at a time, in the order given, and only choose from slots that remain unassigned at that point.
4. Each client has a requested number of sessions in the "sessions" field, and stated day/time availability — schedule exactly that many sessions per client where possible, within their stated availability.
5. If a client's requested sessions cannot all be scheduled within their availability and remaining open slots, schedule as many as possible and clearly note which requested sessions could not be fulfilled and why.
6. Before finalizing, review your full assignment list and confirm no two sessions share the same day and start time. If you find a conflict, correct it and re-check.
7. After your step-by-step reasoning, output the final schedule under a line that reads exactly:
FINAL SCHEDULE:
followed by one session per line in this exact format:
Name | Day | StartTime | EndTime
Schedule this weeks appointments as per below client availability data. ```{client_availability_list}```
"""

REFLECTION_TEMPLATE = """
Review the schedule below for any conflicts — two clients booked at the
same day and time, or sessions outside trainer availability.
If conflicts exist, fix them and return the corrected full schedule.
If no conflicts exist, return the schedule unchanged.

Schedule:
{draft_schedule}
"""

# scheduling_prompt = ChatPromptTemplate.from_template(SCHEDULING_TEMPLATE)
client = Client()
scheduling_prompt = client.pull_prompt("scheduler-prompt")

