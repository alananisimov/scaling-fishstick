from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bank_ctf.data import DatasetSpec, generate_dataset


@dataclass
class CheckResult:
    stage: int
    name: str
    ok: bool
    messages: list[str] = field(default_factory=list)
    flag: str | None = None


def run_cmd(cmd: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)


def fail(stage: int, name: str, msg: str) -> CheckResult:
    return CheckResult(stage, name, False, [msg])


def green_flag(stage: int) -> str:
    return f"GREENBANK{{stage{stage}_open_checker_pass}}"


def require_ok(proc: subprocess.CompletedProcess[str], context: str) -> str | None:
    if proc.returncode != 0:
        return f"{context} failed with code {proc.returncode}: stdout={proc.stdout[-800:]} stderr={proc.stderr[-1200:]}"
    return None


def assert_prediction_keys(pred: pd.DataFrame, key: str, expected: Any, value_col: str) -> str | None:
    if list(pred.columns) != [key, value_col]:
        return f"prediction columns must be {[key, value_col]}, got {list(pred.columns)}"
    if pred[key].duplicated().any():
        return f"prediction file has duplicate {key}"
    if set(pred[key]) != set(expected):
        return f"prediction keys do not match expected {key} set"
    vals = pred[value_col].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        return f"{value_col} contains NaN or inf"
    return None


def make_public_dataset(root: Path, seed: int = 20261101, train_clients: int = 320, test_clients: int = 123) -> Path:
    data = root / "data"
    generate_dataset(data, DatasetSpec(seed=seed, train_clients=train_clients, test_clients=test_clients, train_branches=12, test_branches=5))
    return data


def stage1(repo: Path, work: Path) -> CheckResult:
    data = make_public_dataset(work, seed=101)
    art = work / "artifacts" / "inflow"
    pred_path = work / "inflow_predictions.csv"
    cmd = [sys.executable, "-m", "bank_ctf", "train", "--task", "inflow", "--config", str(repo / "configs/inflow.json"), "--data-dir", str(data), "--output-dir", str(art)]
    err = require_ok(run_cmd(cmd, repo), "stage1 train inflow")
    if err:
        return fail(1, "inflow train/predict contract", err)
    if not (art / "inflow_model.joblib").exists() or not (art / "manifest.json").exists():
        return fail(1, "inflow train/predict contract", "missing inflow_model.joblib or manifest.json")
    cmd = [sys.executable, "-m", "bank_ctf", "predict", "--task", "inflow", "--model-dir", str(art), "--data-dir", str(data), "--output", str(pred_path)]
    err = require_ok(run_cmd(cmd, repo), "stage1 predict inflow")
    if err:
        return fail(1, "inflow train/predict contract", err)
    pred = pd.read_csv(pred_path)
    test = pd.read_csv(data / "clients_test.csv")
    msg = assert_prediction_keys(pred, "client_id", test["client_id"], "predicted_inflow")
    if msg:
        return fail(1, "inflow train/predict contract", msg)
    if (pred["predicted_inflow"] < 0).any():
        return fail(1, "inflow train/predict contract", "predicted_inflow must be non-negative")

    shuffled = data / "shuffled"
    shutil.copytree(data, shuffled)
    clients_test = pd.read_csv(shuffled / "clients_test.csv").sample(frac=1.0, random_state=7).reset_index(drop=True)
    clients_test.to_csv(shuffled / "clients_test.csv", index=False)
    pred2 = work / "inflow_predictions_shuffled.csv"
    err = require_ok(run_cmd([sys.executable, "-m", "bank_ctf", "predict", "--task", "inflow", "--model-dir", str(art), "--data-dir", str(shuffled), "--output", str(pred2)], repo), "stage1 predict shuffled inflow")
    if err:
        return fail(1, "inflow train/predict contract", err)
    pred_shuf = pd.read_csv(pred2)
    if list(pred_shuf["client_id"]) != list(clients_test["client_id"]):
        return fail(1, "inflow train/predict contract", "prediction output order must follow input clients_test order")
    return CheckResult(1, "inflow train/predict contract", True, ["stage 1 public checks passed; hidden checker will score MAE on organizer data"], green_flag(1))


def stage_not_implemented(stage: int) -> CheckResult:
    return fail(stage, f"stage {stage} full contract", "starter template only implements the stage-1 inflow baseline; implement this stage before expecting it to pass")


STAGES = {
    1: stage1,
    2: lambda repo, work: stage_not_implemented(2),
    3: lambda repo, work: stage_not_implemented(3),
    4: lambda repo, work: stage_not_implemented(4),
    5: lambda repo, work: stage_not_implemented(5),
}


def run_stages(repo: Path, stages: list[int]) -> list[CheckResult]:
    results: list[CheckResult] = []
    with tempfile.TemporaryDirectory(prefix="greenbank_public_checker_") as td:
        base = Path(td)
        for stage in stages:
            stage_work = base / f"stage{stage}"
            stage_work.mkdir()
            try:
                results.append(STAGES[stage](repo, stage_work))
            except Exception as exc:
                results.append(fail(stage, f"stage {stage}", f"checker exception {type(exc).__name__}: {exc}"))
    return results


def print_report(results: list[CheckResult], json_path: Path | None = None) -> int:
    payload: list[dict[str, Any]] = []
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] stage {r.stage}: {r.name}")
        for msg in r.messages:
            print(f"  - {msg}")
        if r.flag:
            print(f"  flag: {r.flag}")
        payload.append({"stage": r.stage, "name": r.name, "ok": r.ok, "messages": r.messages, "flag": r.flag})
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if all(r.ok for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Public starter checker for Green Bank Pipeline CTF")
    p.add_argument("--repo", default=".", help="participant repository root")
    p.add_argument("--stage", type=int, choices=range(1, 6), action="append", help="stage to run; repeat or omit for all")
    p.add_argument("--json-report", type=Path)
    args = p.parse_args(argv)
    repo = Path(args.repo).resolve()
    stages = args.stage or [1]
    return print_report(run_stages(repo, stages), args.json_report)


if __name__ == "__main__":
    raise SystemExit(main())
