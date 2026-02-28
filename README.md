# 🗓️ AI-Powered Client Appointment Scheduler
This project demonstrates how to build an **AI-assisted appointment scheduling system** using:
- **OpenAI GPT-4.1-nano**
- **Function calling for structured output**
- **Pandas** for CSV handling
- **Python** for workflow automation

The script reads client availability from a CSV file, considers trainer constraints, and generates a weekly appointment schedule without time conflicts.

---

## 📌 Features
- Reads **client availability** from `availability.csv`
- Defines **trainer availability rules** as part of system prompt
- Uses OpenAI's **function calling** for consistent structured output
- Exports results to `scheduled_sessions.csv`

---

## 🏛️ Architectural Decisions

### 1. CSV as the Data Layer
Rather than using a database or live calendar API, this solution intentionally uses **processed data in CSV format** as its input layer. This keeps the project lightweight, portable, and easy to inspect or modify without additional infrastructure. The `availability.csv` file acts as a clean, pre-processed snapshot of client availability that the LLM can reason over reliably.

### 2. System Prompt as the Rule Engine
Trainer availability and scheduling constraints are defined directly in the **system prompt** rather than hard-coded in application logic. This is a deliberate architectural choice — it keeps the rules human-readable, easy to update, and decoupled from the code itself. Changing a trainer's working hours or break windows requires no code changes, just a prompt edit.

### 3. Structured Output via Function Calling
OpenAI's **function calling** feature is used to enforce a consistent, parseable output schema from the model. This avoids brittle text parsing and ensures the generated schedule can be reliably written to CSV regardless of how the model phrases its response.

### 4. Reflection + LLM-as-Judge for Evaluation
A **reflection-based evaluation** pattern is used to validate the generated schedule. After the initial schedule is produced, a second LLM call acts as an **independent judge**, reviewing the output for:
- **Time conflicts** — overlapping appointments for the same trainer or client
- **Missing sessions** — required clients who were not scheduled

This self-critique loop improves output quality without requiring ground truth labels or manual review, making it a practical evaluation strategy for generative scheduling tasks.

---

## 🛠️ Requirements

Install dependencies:
```bash
pip install openai python-dotenv pandas
```

## Want to try?
- Use your own OpenAI API Key