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
trainer_availability = {
            "Morning": [("06:00", "09:00")],
            "Evening": [("17:00", "21:00")]
        }

# Prepare the prompt
sys_prompt = f"""
Act as an assistant to schedule appointments for clients for upcoming week. Starting from Monday to Saturday.
Follow below rules while scheduling:
1. Each session is 1 hour in duration. Give start and end time for all appointments.
2. Schedule only within trainer's availability ```{trainer_availability}```
3. No two clients should have sessions scheduled at the same time.
4. Trainer is not available on Thursday and Saturday evenings.
5. Trainer is not available on any time on a Sunday.
"""

user_prompt = f"""
Please schedule this weeks appointments as per below client availability
Schedule as many sessions as instructed in the sessions column ```{client_availability_list}```
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

result = []
for item in sessions:
    result.append(json.loads(item))

df = pd.DataFrame(result)
df.to_csv("scheduled_sessions.csv", index=False)