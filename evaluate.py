from langsmith.evaluation import evaluate
from evaluators import no_conflict_evaluator, session_count_evaluator
from scheduling_chain import build_chain

chain = build_chain()

evaluate(
    lambda inputs: chain.invoke(inputs),
    data="scheduler-dataset",
    evaluators=[no_conflict_evaluator, session_count_evaluator],
    experiment_prefix="scheduler-eval",
)