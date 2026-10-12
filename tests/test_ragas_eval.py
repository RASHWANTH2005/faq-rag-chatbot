import pytest
from src.evaluation.run_eval import run_evaluation

@pytest.mark.live_eval
def test_ragas_evaluation():
    df = run_evaluation()

    assert not df.empty

    # Column names come from the metric names
    faith = df["faithfulness"]
    prec = df["llm_context_precision_with_reference"]

    # Fail loudly if the judge errored out on records (those become NaN)
    assert faith.notna().all(), f"NaN faithfulness at rows {faith[faith.isna()].index.tolist()}"
    assert prec.notna().all(), f"NaN precision at rows {prec[prec.isna()].index.tolist()}"

    # Aggregate thresholds (mean over all records, not per batch)
    assert faith.mean() >= 0.80
    assert prec.mean() >= 0.70

    # bad = df[(df["faithfulness"] < 0.5)]
    # assert bad.empty, bad[["user_input", "faithfulness"]]