"""
main.py
-------
Entry point. Loads data, runs the scheduling chain, writes results to CSV.
"""

import pandas as pd
from data_loader import load_trainer_slots, load_client_availability, to_json
from scheduling_chain import build_chain
from config import OUTPUT_CSV_PATH

def main() -> None:
    available_slots = load_trainer_slots()
    client_availability_list = load_client_availability()

    chain = build_chain()
    response = chain.invoke({
        "available_slots": to_json(available_slots),
        "client_availability_list": to_json(client_availability_list),
    })

    df = pd.DataFrame([session.model_dump() for session in response.sessions])
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Schedule written to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
