"""
data_loader.py
--------------
All I/O for reading the trainer's availability and the client request data.
"""

import json
from typing import List, Dict, Any

import pandas as pd

from utils import available_slot_generator
from config import AVAILABILITY_CSV_PATH


def load_trainer_slots() -> List[Any]:
    """Return the trainer's available 1-hour slots for the week."""
    return available_slot_generator()


def load_client_availability(csv_path: str = AVAILABILITY_CSV_PATH) -> List[Dict[str, Any]]:
    """Read the client availability CSV into a list of plain dicts.

    Each dict has keys: name, day, time, sessions (requested session count).
    """
    df = pd.read_csv(csv_path)
    return [
        {
            "name": row["name"],
            "day": row["day"],
            "time": row["time"],
            "sessions": row["sessions"],
        }
        for _, row in df.iterrows()
    ]


def to_json(data: Any) -> str:
    """Pretty-print helper used when injecting data into prompt templates."""
    return json.dumps(data, indent=2)
