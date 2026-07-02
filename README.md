# OpenAI Session Scheduler

An LLM-powered weekly session scheduler for personal trainers. Given a trainer's availability and client requests, it produces a conflict-free schedule — with automatic conflict detection and self-reflection if the model makes a mistake.

## How it works

1. Trainer availability and client requests are loaded from CSV
2. A scheduling prompt is sent to `gpt-4o-mini` via LangChain
3. The output is parsed and checked for double-bookings deterministically
4. If a conflict is found, the LLM is re-invoked with a reflection prompt to fix it
5. The corrected schedule is coerced into a validated Pydantic model and saved to CSV

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Orchestration | LangChain |
| Structured output | Pydantic v2 |
| Observability | LangSmith (tracing, datasets, evaluations, prompt hub) |
| Data | pandas |

## Project Structure

```
OpenAI_Session_Scheduler/
├── config.py               # Model settings, file paths, LLM client factory
├── models.py               # Pydantic schemas: Session, Schedule
├── data_loader.py          # Loads trainer slots + client availability CSV
├── prompts.py              # Scheduling and reflection prompt templates
├── conflict.py             # Parses LLM output, detects scheduling conflicts
├── scheduling_chain.py     # LangChain pipeline: prompt → LLM → reflect → structured output
├── main.py                 # Entry point
├── evaluate.py             # LangSmith evaluation runner (run separately)
├── evaluators.py           # Custom evaluator functions
├── utils.py                # available_slot_generator, print_reasoning
├── availability.csv        # Client availability input
└── scheduled_sessions.csv  # Generated schedule output
```

## Observability with LangSmith

Every run is fully traced in LangSmith — prompts, token usage, latency, and whether the reflection step fired.

**Evaluators** run against a saved dataset of real runs:

| Evaluator | What it checks |
|---|---|
| `no_conflict_evaluator` | No two sessions share the same day and start time |
| `session_count_evaluator` | Output contains at least one scheduled session |

**Prompt versioning** — the scheduling prompt is stored and versioned in LangSmith Prompt Hub. Pull the latest version at runtime:
```python
scheduling_prompt = Client().pull_prompt("scheduler-prompt")
```

Run evals independently of production:
```bash
python evaluate.py   # scores all dataset examples
python main.py       # production run only
```

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/your-username/OpenAI_Session_Scheduler.git
cd OpenAI_Session_Scheduler
pip install -r requirements.txt
```

**2. Configure environment**

Create a `.env` file in the project root:
```
# OpenAI
OPENAI_API_KEY=sk-your-openai-key-here

# LangSmith
LANGCHAIN_TRACING=true
LANGCHAIN_API_KEY=lsv2_pt_your-langsmith-key-here
LANGCHAIN_PROJECT=scheduler-assistant
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

> Remove `LANGSMITH_ENDPOINT` if you are on the US region.

**3. Add client availability**

Populate `availability.csv` with client requests:
```
name,day,time,sessions
Alice,Monday,Morning,2
Bob,Tuesday,Evening,3
```

**4. Run**
```bash
python main.py
```

Output is written to `scheduled_sessions.csv`.

## Requirements

```
langchain
langchain-openai
langchain-core
langsmith
pydantic
pandas
python-dotenv
```

## Design notes

- Implement reasoning and output formatting as individual steps in the chain utilizing Langchain framework. To offload formatting from LLM's thinking loop and ensure focus on scheduling without conflict
- Define output format in pydantic model rather than descriptive function call for reuse of the structure
- Perform deterministic conflict check to avoid reflection call to LLM if no conflict found
- Version controlled scheduling prompt to avoid code changes for prompt development and capability to rollback prompt changes when evaluation fails
- Langsmith Observability implemented to track accuracy, token usage and maintain example datasets