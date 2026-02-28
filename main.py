from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read the Client availability for the week
client_availability = pd.read_csv("availability.csv")

# Extract details from csv file
client_availability_list = []
for index, row in client_availability.iterrows():
    row_dict = {"name": row["name"], "day": row["day"], "time": row["time"], "sessions": row["sessions"]}
    client_availability_list.append(row_dict)

# Trainer's availability for the week (Hard-coded)
trainer_availability ={
    "Morning": [
        {"start": "06:00", "end": "09:00"}
    ],
    "Evening": [
        {"start": "17:00", "end": "21:00"}
    ]
}

scheduling_restrictions = f"""
1. Session duration: 1 hour
2. Schedule only within trainer's availability: 
   {trainer_availability}
Morning and Evening are time windows. Sessions must be scheduled only within these windows. Use exactly start and end keys.
3. No two clients can have sessions at the same time and same day.
4. Trainer unavailable:
   - Thursday & Saturday evenings
   - Sunday any time
5. Think step by step: generate all possible slots, remove conflicts, assign sessions, then output.
6. Use the tool provided to format the output and avoid any commentary or additional content in the response.
"""

# Prepare the prompt
sys_prompt = f"""
Act as an assistant to schedule appointments for clients for upcoming week. Starting from Monday to Saturday.
Follow below rules while scheduling:
{scheduling_restrictions}
"""

user_prompt = f"""
Please schedule this weeks appointments as per below client availability
Schedule as many sessions as instructed in the sessions column ```{client_availability_list}```
"""

sys_prompt_for_reflection = f"""
Act as a reviewer to review scheduled client sessions, look for conflicting session timings and fix them. 
Schedule should follow below restrictions and trainer availability:
{scheduling_restrictions}
Give special thought about below constraints which reviewing and fixing the schedule.
1. Check if any 2 clients have sessions scheduled on same day and same time. If found, reschedule at a suitable time
2. Check if any session falls out of trainer availability. If so, reschedule at a suitable time
3. Check if schedule is missing for any client. If so, schedule for them as per the rules
"""

# Response formatting by function calling
response_formatter = {
        "type": "function",
        "function": {
            "name": "response_formatter",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Name of Client"},
                    "session_day": { "type": "string", "description": "Day of the scheduled session"},
                    "start_time": { "type": "string", "description": "Start time of the scheduled session"},
                    "end_time": { "type": "string", "description": "End time of the scheduled session"}
                }
            }
        }
    }

# Call OpenAI API
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ],
    tools = [response_formatter]
)

sessions = [i.function.arguments for i in response.choices[0].message.tool_calls]

user_prompt_for_reflection = f"""
Can you review below schedule and fix the conflicting session timings.
{sessions}
Respond the new schedule in JSON format as per tools provided.
Here is the clients availability on each day of the week:
```{client_availability_list}```
"""

# Call OpenAI API for reflection
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": sys_prompt_for_reflection},
        {"role": "user", "content": user_prompt_for_reflection}
    ],
    tools = [response_formatter]
)
# Result after first attempt
result = []
for item in sessions:
    result.append(json.loads(item))

df = pd.DataFrame(result)
df.to_csv("scheduled_sessions.csv", index=False)

# Result after reflection
sessions_after_reflection = [i.function.arguments for i in response.choices[0].message.tool_calls]
result_after_reflection = []
for item in sessions_after_reflection:
    result_after_reflection.append(json.loads(item))

df = pd.DataFrame(result_after_reflection)
df.to_csv("scheduled_sessions_after_reflection.csv", index=False)