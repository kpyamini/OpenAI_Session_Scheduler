from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import json

from openai.types.beta.realtime.transcription_session_update_param import Session

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client_availability = pd.read_csv("availability.csv")
client_availability_list = []
for index, row in client_availability.iterrows():
    row_dict = {"name": row["name"], "day": row["day"], "time": row["time"], "sessions": row["sessions"]}
    client_availability_list.append(row_dict)

my_availability = {
            "Morning": [("06:00", "09:00")],
            "Evening": [("17:00", "21:00")]
        }

# Create prompt for OpenAI
sys_prompt = f"""
Act as an assistant to schedule appointments for my clients for upcoming week. Starting from Monday to Saturday.
Each session is 1 hour in duration. Give start and end time for all appointments.
My availability is as per below list.
{my_availability}
Make sure there is no clashing timings between clients.
Do not schedule appointments apart from the available time mentioned.
I am not available on Thursday and Saturday evenings.
I am not available on any time on a Sunday.
"""
user_prompt = f"""
Please schedule this weeks appointments as per below client availability
Schedule as may sessions as instructed in the sessions column
{client_availability_list}
"""

# Call OpenAI API
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ],
    tools = [{
        "type": "function",
        "function": {
            "name": "Json_formatter",
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
    }]
)

sessions = [i.function.arguments for i in response.choices[0].message.tool_calls]

result = []
for item in sessions:
    result.append(json.loads(item))

df = pd.DataFrame(result)
df.to_csv("sessions_schedule.csv", index=False)