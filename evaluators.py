
def no_conflict_evaluator(run, example):
    """Check no two sessions share the same day and time."""
    sessions = run.outputs.get("sessions", [])

    seen = set()
    for session in sessions:
        key = (session["session_day"], session["start_time"])
        if key in seen:
            return {"key": "no_conflicts", "score": 0}
        seen.add(key)

    return {"key": "no_conflicts", "score": 1}


def session_count_evaluator(run, example):
    """Check output session count is greater than zero."""
    sessions = run.outputs.get("sessions", [])
    return {
        "key": "has_sessions",
        "score": 1 if len(sessions) > 0 else 0
    }