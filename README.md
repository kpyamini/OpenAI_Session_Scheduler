# Scheduling Assistant

A LangChain pipeline that turns a trainer's availability + a list of
client requests into a conflict-free weekly schedule, using an LLM for
the assignment and a deterministic check for conflict detection.

## Structure

```
scheduler/
├── config.py             # Model settings, file paths, client factory
├── models.py             # Pydantic schemas: Session, Schedule
├── data_loader.py        # Reads trainer slots + client availability CSV
├── prompts.py             # Prompt templates (scheduling + reflection)
├── conflict.py            # Parses LLM text output, detects scheduling conflicts
├── scheduling_chain.py     # Wires prompt -> LLM -> reflection -> structured output
├── main.py                # Entry point: orchestrates the above
└── utils.py                # (unchanged) available_slot_generator, print_reasoning
```

## Why split it up this way

Think of the original single file as one cook running every station in a
kitchen — taking orders, chopping, cooking, and plating all at once. Works
fine for one dish, but a mistake or a needed change in one step risks
knocking into the next. Splitting into modules gives each step its own
station:

| Module | Responsibility | Depends on LLM? |
|---|---|---|
| `config.py` | "knobs" — model name, temperature, paths | No |
| `models.py` | shape of the data | No |
| `data_loader.py` | reading inputs | No |
| `prompts.py` | wording sent to the model | No |
| `conflict.py` | parsing + validating model output | No |
| `scheduling_chain.py` | gluing the above into a LangChain pipeline | Yes |
| `main.py` | running it end to end | Yes (via the chain) |

Five of the seven modules have **no dependency on the LLM at all** — they're
plain Python you can unit test directly, e.g.:

```python
from conflict import has_conflict

def test_detects_duplicate_slot():
    draft = """
    FINAL SCHEDULE:
    Alice | Monday | 09:00 | 10:00
    Bob   | Monday | 09:00 | 10:00
    """
    assert has_conflict(draft) is True
```

That wasn't possible in the original script, since parsing logic, prompt
text, and chain construction were all interleaved in one module-level block
that ran on import.

## Running it

```bash
python main.py
```

This expects `availability.csv` (client requests) in the working directory
and a `.env` file with your OpenAI credentials, same as the original script.
Output is written to `scheduled_sessions_langchain.csv`.

## Notable behavioral notes carried over from the prototype

- The reflection step only fires if `has_conflict()` returns `True` — the
  happy path costs exactly one LLM call, same as before.
- `extract_final_schedule` silently skips lines that don't match the
  `Name | Day | Start | End` shape; it's lenient about reasoning text
  that might contain stray `|` characters before the `FINAL SCHEDULE:`
  marker.
- `conflict.py` raises `ScheduleFormatError` (subclass of `ValueError`)
  if the model's response is missing the `FINAL SCHEDULE:` marker
  entirely — same failure mode as before, just named for clarity.

## Suggested follow-ups (not yet implemented)

- Add a `tests/` directory with unit tests for `conflict.py` and
  `data_loader.py` (no API key needed for either).
- Replace `print()` calls in `conflict.py` with the `logging` module so
  verbosity can be controlled without code changes.
- Consider validating `client_availability_list["sessions"]` as an int
  at load time, to fail fast on malformed CSV data rather than passing
  a bad value through to the prompt.
