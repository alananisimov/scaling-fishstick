from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-not-found]


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def test_generator_contract(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    proc = run([
        sys.executable,
        "-m",
        "bank_ctf.generate_data",
        "--output-dir",
        str(data),
        "--train-clients",
        "180",
        "--test-clients",
        "70",
        "--train-branches",
        "10",
        "--test-branches",
        "4",
    ], repo)
    assert proc.returncode == 0, proc.stderr
    clients_train = pd.read_csv(data / "clients_train.csv")
    clients_test = pd.read_csv(data / "clients_test.csv")
    credit_train = pd.read_csv(data / "credit_train.csv")
    folds_group = pd.read_csv(data / "folds_group.csv")
    assert clients_train["client_id"].is_unique
    assert clients_test["client_id"].is_unique
    assert "inflow_next_30d" not in clients_test.columns
    assert not (data / "clients_test_private.csv").exists()
    assert not (data / "credit_test_private.csv").exists()
    assert set(credit_train["client_id"]).issubset(set(clients_train["client_id"]))
    merged = clients_train[["client_id", "branch_id"]].merge(folds_group, on="client_id")
    assert merged.groupby("branch_id")["fold_id"].nunique().max() == 1


def test_stage1_baseline_train_predict(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    proc = run([sys.executable, "-m", "bank_ctf.generate_data", "--output-dir", str(data)], repo)
    assert proc.returncode == 0, proc.stderr

    model_dir = tmp_path / "inflow_model"
    pred_path = tmp_path / "inflow_predictions.csv"
    proc = run([
        sys.executable,
        "-m",
        "bank_ctf",
        "train",
        "--task",
        "inflow",
        "--config",
        "configs/inflow.json",
        "--data-dir",
        str(data),
        "--output-dir",
        str(model_dir),
    ], repo)
    assert proc.returncode == 0, proc.stderr
    assert (model_dir / "inflow_model.joblib").exists()
    assert (model_dir / "manifest.json").exists()

    proc = run([
        sys.executable,
        "-m",
        "bank_ctf",
        "predict",
        "--task",
        "inflow",
        "--model-dir",
        str(model_dir),
        "--data-dir",
        str(data),
        "--output",
        str(pred_path),
    ], repo)
    assert proc.returncode == 0, proc.stderr
    preds = pd.read_csv(pred_path)
    clients_test = pd.read_csv(data / "clients_test.csv")
    assert list(preds.columns) == ["client_id", "predicted_inflow"]
    assert preds["client_id"].tolist() == clients_test["client_id"].tolist()
    assert bool(preds["predicted_inflow"].notna().all())
    assert bool((preds["predicted_inflow"] >= 0).all())


def test_later_stages_are_explicit_todo(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    proc = run([sys.executable, "-m", "bank_ctf.generate_data", "--output-dir", str(data)], repo)
    assert proc.returncode == 0, proc.stderr
    proc = run([
        sys.executable,
        "-m",
        "bank_ctf",
        "train",
        "--task",
        "risk",
        "--config",
        "configs/risk.json",
        "--data-dir",
        str(data),
        "--output-dir",
        str(tmp_path / "risk_model"),
    ], repo)
    assert proc.returncode == 2
    assert "TODO: implement train --task risk beyond the stage-1 inflow baseline." in proc.stderr
