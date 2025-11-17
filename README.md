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
- Uses OpenAI’s **function calling** for consistent structured output
- Exports results to `scheduled_sessions.csv`

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install openai python-dotenv pandas
```

## Want to try ??

- Use your own OpenAI API Key
