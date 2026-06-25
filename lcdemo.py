from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import pandas as pd
import json
from pydantic import BaseModel, Field
from typing import List

from utils import available_slot_generator, print_reasoning


class Session(BaseModel):
    name: str = Field(description="Name of Client")
    session_day: str = Field(description="Day of the scheduled session")
    start_time: str = Field(description="Start time of the scheduled session")
    end_time: str = Field(description="End time of the scheduled session")

class Schedule(BaseModel):
    sessions: List[Session]

load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=2000, timeout=60)

# Trainer's availability for the week (Hard-coded)
available_slots = available_slot_generator()

# Read the Client availability for the week
client_availability = pd.read_csv("availability.csv")

# Extract details from csv file
client_availability_list = []
for index, row in client_availability.iterrows():
    row_dict = {"name": row["name"], "day": row["day"], "time": row["time"], "sessions": row["sessions"]}
    client_availability_list.append(row_dict)

template_string = """
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

prompt_template = ChatPromptTemplate.from_template(template_string)

initial_chain = prompt_template | chat | StrOutputParser() | RunnableLambda(print_reasoning)

def extract_final_schedule(draft_response: str) -> list[dict]:
    if "FINAL SCHEDULE:" not in draft_response:
        raise ValueError("Schedule not found in expected format, cannot check conflicts")

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

def check_conflict(draft_response: str) -> bool:
    sessions_to_review = extract_final_schedule(draft_response)

    seen = {}
    conflicts = []
    for session in sessions_to_review:
        key = (session["session_day"], session["start_time"])
        if key in seen:
            conflicts.append(key)
        else:
            seen[key] = session

    if conflicts:
        print("Initial schedule has conflicts! Proceeding to reflect and fix the schedule")
        return True
    else:
        print("Initial schedule is accurate! No reflection required")
        return False

def check_and_reflect(draft_response: str) -> str:
    if not check_conflict(draft_response):
        return draft_response

    reflection_prompt = f"""
        Review the schedule below for any conflicts — two clients booked at the 
        same day and time, or sessions outside trainer availability.
        If conflicts exist, fix them and return the corrected full schedule.
        If no conflicts exist, return the schedule unchanged.
    
        Schedule:
        {draft_response}
        """
    reflection_response = chat.invoke(reflection_prompt)
    return reflection_response.content

structured_chat = chat.with_structured_output(Schedule)

final_chain = initial_chain | RunnableLambda(check_and_reflect) | structured_chat

response = final_chain.invoke({
    "available_slots": json.dumps(available_slots, indent=2),
    "client_availability_list": json.dumps(client_availability_list, indent=2)
})

df = pd.DataFrame([session.model_dump() for session in response.sessions])
df.to_csv("scheduled_sessions_langchain.csv", index=False)