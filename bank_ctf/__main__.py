from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CLIENT_FEATURES

FEATURE_COLUMNS = ["branch_id", *CLIENT_FEATURES]
CATEGORICAL = ["branch_id", "region_code"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bank_ctf", description="Green Bank Pipeline CTF participant CLI")
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--task", choices=["inflow", "risk"], required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--task", choices=["inflow", "risk"], required=True)
    pred.add_argument("--model-dir", required=True)
    pred.add_argument("--data-dir", required=True)
    pred.add_argument("--output", required=True)
    exp = sub.add_parser("experiment")
    exp.add_argument("--config", required=True)
    exp.add_argument("--data-dir", required=True)
    exp.add_argument("--runs-dir", required=True)
    exp.add_argument("--run-id", required=True)
    return p


def read_config(path: str | Path) -> dict[str, Any]:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {path}: {exc}") from exc
    if cfg.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    return cfg


def make_inflow_model(cfg: dict[str, Any]) -> Pipeline:
    model_spec = cfg.get("model", {})
    params = dict(model_spec.get("params", {}))
    seed = int(cfg.get("seed", 0))
    kind = model_spec.get("kind", "hist_gradient_boosting")
    if kind == "hist_gradient_boosting":
        allowed = {"max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization", "min_samples_leaf"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported inflow params: {sorted(unknown)}")
        estimator = HistGradientBoostingRegressor(random_state=seed, **params)
    elif kind == "ridge":
        allowed = {"alpha"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported inflow params: {sorted(unknown)}")
        estimator = Ridge(random_state=seed, **params)
    else:
        raise ValueError(f"unsupported inflow model kind: {kind}")

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def require_columns(df: pd.DataFrame, columns: list[str], file_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} missing columns: {missing}")


def train_inflow(config_path: str, data_dir: str, output_dir: str) -> None:
    cfg = read_config(config_path)
    if cfg.get("task") != "inflow":
        raise ValueError("train --task inflow requires config task='inflow'")
    data = Path(data_dir)
    train = pd.read_csv(data / "clients_train.csv")
    require_columns(train, ["client_id", "inflow_next_30d", *FEATURE_COLUMNS], "clients_train.csv")
    if not train["client_id"].is_unique:
        raise ValueError("clients_train.csv has duplicate client_id")
    model = make_inflow_model(cfg)
    model.fit(train[FEATURE_COLUMNS], train["inflow_next_30d"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "config": cfg}, out / "inflow_model.joblib")
    manifest = {"task": "inflow", "model_file": "inflow_model.joblib", "feature_columns": FEATURE_COLUMNS, "effective_config": cfg}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def predict_inflow(model_dir: str, data_dir: str, output: str) -> None:
    bundle = joblib.load(Path(model_dir) / "inflow_model.joblib")
    clients = pd.read_csv(Path(data_dir) / "clients_test.csv")
    require_columns(clients, ["client_id", *bundle["feature_columns"]], "clients_test.csv")
    if not clients["client_id"].is_unique:
        raise ValueError("clients_test.csv has duplicate client_id")
    preds = np.maximum(0, np.asarray(bundle["model"].predict(clients[bundle["feature_columns"]]), dtype=float))
    if not np.isfinite(preds).all():
        raise ValueError("model produced non-finite predictions")
    out = pd.DataFrame({"client_id": clients["client_id"], "predicted_inflow": preds})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train" and args.task == "inflow":
            train_inflow(args.config, args.data_dir, args.output_dir)
            return 0
        if args.command == "predict" and args.task == "inflow":
            predict_inflow(args.model_dir, args.data_dir, args.output)
            return 0
        print(
            f"TODO: implement {args.command} --task {getattr(args, 'task', '<from config>')} beyond the stage-1 inflow baseline.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
