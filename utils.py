from datetime import datetime, timedelta
from typing import List, Dict

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
windows = {"Morning": ("06:00", "09:00"), "Evening": ("17:00", "21:00")}
unavailable = {"Thursday": ["Evening"], "Saturday": ["Evening"]}  # Sunday excluded entirely already

def hourly_slots(start, end):
    fmt = "%H:%M"
    t = datetime.strptime(start, fmt)
    end_t = datetime.strptime(end, fmt)
    slots = []
    while t + timedelta(hours=1) <= end_t:
        slots.append((t.strftime(fmt), (t + timedelta(hours=1)).strftime(fmt)))
        t += timedelta(hours=1)
    return slots

def available_slot_generator() -> List[Dict]:
    """Generate availability slot in hour windows for all applicable days in the week.
    """
    available_slots = []
    for day in days:
        for window_name, (start, end) in windows.items():
            if window_name in unavailable.get(day, []):
                continue
            for s, e in hourly_slots(start, end):
                available_slots.append({"day": day, "start": s, "end": e})
    return available_slots

def print_reasoning(response: str) -> str:
    print(response)
    print("___END OF REASONING RESPONSE___")
    return response
