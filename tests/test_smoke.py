import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.smoke
def test_batch_entrypoint_runs_and_writes_results():
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "scored.csv"
    if output_path.exists():
        output_path.unlink()

    completed = subprocess.run(
        [sys.executable, "-m", "fraud_service.batch"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()

    scored = pd.read_csv(output_path)
    assert {"transaction_id", "score", "decision"}.issubset(scored.columns)
    assert len(scored) == 5000
    assert set(scored["decision"]).issubset({"allow", "review", "block"})
    assert "scoring_complete" in completed.stderr
