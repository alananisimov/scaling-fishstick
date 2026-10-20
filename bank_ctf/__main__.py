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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CLIENT_FEATURES, CREDIT_FEATURES

FEATURE_COLUMNS = ["branch_id", *CLIENT_FEATURES]
CATEGORICAL = ["branch_id", "region_code"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]

CATEGORICAL_RISK = ["application_id", "client_id"]
RISK_FEATURE_COLUMNS = ["application_id", "client_id", *CREDIT_FEATURES]
NUMERIC_RISK = [c for c in RISK_FEATURE_COLUMNS if c not in CATEGORICAL_RISK]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bank_ctf", description="Green Bank Pipeline CTF participant CLI"
    )
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
        allowed = {
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "l2_regularization",
            "min_samples_leaf",
        }
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

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def make_risk_model(cfg: dict[str, Any], extra_numeric=()) -> Pipeline:
    model_spec = cfg.get("model", {})
    params = dict(model_spec.get("params", {}))
    seed = int(cfg.get("seed", 0))
    kind = model_spec.get("kind", "hist_gradient_boosting")
    if kind == "hist_gradient_boosting":
        allowed = {
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "l2_regularization",
            "min_samples_leaf",
        }
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported risk params: {sorted(unknown)}")
        estimator = HistGradientBoostingClassifier(random_state=seed, **params)
    elif kind == "logistic_regression":
        allowed = {"C", "max_iter"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported risk params: {sorted(unknown)}")
        estimator = LogisticRegression(random_state=seed, **params)
    else:
        raise ValueError(f"unsupported risk model kind: {kind}")

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                [*NUMERIC_RISK, *extra_numeric],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_RISK,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def train_risk(config_path: str, data_dir: str, output_dir: str) -> None:
    cfg = read_config(config_path)
    if cfg.get("task") != "risk":
        raise ValueError("train --task risk requires config task='risk'")
    data = Path(data_dir)
    train = pd.read_csv(data / "credit_train.csv")
    require_columns(
        train,
        ["application_id", "overdue_next_60d", *RISK_FEATURE_COLUMNS],
        "credit_train.csv",
    )
    if not train["application_id"].is_unique:
        raise ValueError("credit_train.csv has duplicate application_id")
    model = make_risk_model(cfg)
    model.fit(train[RISK_FEATURE_COLUMNS], train["overdue_next_60d"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": RISK_FEATURE_COLUMNS, "config": cfg},
        out / "risk_model.joblib",
    )
    manifest = {
        "task": "risk",
        "model_file": "risk_model.joblib",
        "feature_columns": RISK_FEATURE_COLUMNS,
        "effective_config": cfg,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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
    require_columns(
        train, ["client_id", "inflow_next_30d", *FEATURE_COLUMNS], "clients_train.csv"
    )
    if not train["client_id"].is_unique:
        raise ValueError("clients_train.csv has duplicate client_id")
    model = make_inflow_model(cfg)
    model.fit(train[FEATURE_COLUMNS], train["inflow_next_30d"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS, "config": cfg},
        out / "inflow_model.joblib",
    )
    manifest = {
        "task": "inflow",
        "model_file": "inflow_model.joblib",
        "feature_columns": FEATURE_COLUMNS,
        "effective_config": cfg,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def predict_inflow(model_dir: str, data_dir: str, output: str) -> None:
    bundle = joblib.load(Path(model_dir) / "inflow_model.joblib")
    clients = pd.read_csv(Path(data_dir) / "clients_test.csv")
    require_columns(
        clients, ["client_id", *bundle["feature_columns"]], "clients_test.csv"
    )
    if not clients["client_id"].is_unique:
        raise ValueError("clients_test.csv has duplicate client_id")
    preds = np.maximum(
        0,
        np.asarray(
            bundle["model"].predict(clients[bundle["feature_columns"]]), dtype=float
        ),
    )
    if not np.isfinite(preds).all():
        raise ValueError("model produced non-finite predictions")
    out = pd.DataFrame({"client_id": clients["client_id"], "predicted_inflow": preds})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def predict_risk(model_dir: str, data_dir: str, output: str) -> None:
    bundle = joblib.load(Path(model_dir) / "risk_model.joblib")
    clients = pd.read_csv(Path(data_dir) / "credit_test.csv")
    require_columns(
        clients, ["application_id", *bundle["feature_columns"]], "credit_test.csv"
    )
    if not clients["application_id"].is_unique:
        raise ValueError("credit_test.csv has duplicate application_id")
    preds = np.maximum(
        0,
        np.asarray(
            bundle["model"].predict(clients[bundle["feature_columns"]]), dtype=float
        ),
    )
    if not np.isfinite(preds).all():
        raise ValueError("model produced non-finite predictions")
    out = pd.DataFrame(
        {"application_id": clients["application_id"], "predicted_risk": preds}
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def experiment(config_path: str, data_dir: str, runs_dir: str, run_id: str):
    cfg = read_config(config_path)
    if cfg.get("task") != "risk" or cfg.get("split_version") != "group_v1":
        raise ValueError("Expected risk / group_v1")

    data = Path(data_dir)
    clients = pd.read_csv(data / "clients_train.csv")
    credit = pd.read_csv(data / "credit_train.csv")
    folds = pd.read_csv(data / "folds_group.csv")

    clients = clients.merge(folds, on="client_id", validate="one_to_one")
    credit = credit.merge(
        clients[["client_id", "fold_id"]], on="client_id", validate="many_to_one"
    )

    if (clients.groupby("branch_id")["fold_id"].nunique() != 1).any():
        raise ValueError("branch_id leakage")
    if clients["fold_id"].isna().any() or credit["fold_id"].isna().any():
        raise ValueError("Missing fold_id")

    inflow_cfg = {**cfg, "model": cfg.get("inflow_model", {})}
    risk_cols = list(
        dict.fromkeys([*RISK_FEATURE_COLUMNS, *CATEGORICAL_RISK, "predicted_inflow"])
    )
    results = []

    for fold in sorted(clients["fold_id"].unique()):
        train = clients[clients.fold_id != fold]
        val = clients[clients.fold_id == fold]

        oof = pd.Series(index=train.client_id, dtype=float)
        for inner in train.fold_id.unique():
            a = train[train.fold_id != inner]
            b = train[train.fold_id == inner]

            model = make_inflow_model(inflow_cfg)
            model.fit(a[FEATURE_COLUMNS], a["inflow_next_30d"])
            oof.loc[b.client_id] = model.predict(b[FEATURE_COLUMNS])

        model = make_inflow_model(inflow_cfg)
        model.fit(train[FEATURE_COLUMNS], train["inflow_next_30d"])
        val_pred = pd.Series(model.predict(val[FEATURE_COLUMNS]), index=val.client_id)

        r_train = credit[credit.fold_id != fold].copy()
        r_val = credit[credit.fold_id == fold].copy()
        r_train["predicted_inflow"] = r_train.client_id.map(oof)
        r_val["predicted_inflow"] = r_val.client_id.map(val_pred)

        model = make_risk_model(cfg, extra_numeric=("predicted_inflow",))
        model.fit(r_train[risk_cols], r_train["overdue_next_60d"])

        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(r_val[risk_cols])[:, 1]
        else:
            pred = np.clip(model.predict(r_val[risk_cols]), 0, 1)

        results.append(
            pd.DataFrame(
                {
                    "application_id": r_val.application_id,
                    "fold_id": fold,
                    "overdue_next_60d": r_val.overdue_next_60d,
                    "predicted_risk": pred,
                }
            )
        )

    out = Path(runs_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(results).to_csv(out / "risk_oof.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train" and args.task == "inflow":
            train_inflow(args.config, args.data_dir, args.output_dir)
            return 0
        if args.command == "predict" and args.task == "inflow":
            predict_inflow(args.model_dir, args.data_dir, args.output)
            return 0
        if args.command == "train" and args.task == "risk":
            train_risk(args.config, args.data_dir, args.output_dir)
            return 0
        if args.command == "predict" and args.task == "risk":
            predict_risk(args.model_dir, args.data_dir, args.output)
            return 0
        if args.command == "experiment":
            experiment(args.config, args.data_dir, args.runs_dir, args.run_id)
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
