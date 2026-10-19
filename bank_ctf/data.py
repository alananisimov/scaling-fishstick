from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

CLIENT_FEATURES = [
    "account_age_days",
    "avg_inflow_30d",
    "avg_inflow_90d",
    "avg_outflow_30d",
    "avg_outflow_90d",
    "balance_mean_30d",
    "balance_std_30d",
    "txn_count_30d",
    "spend_food_share",
    "spend_transport_share",
    "spend_entertainment_share",
    "has_mobile_app",
    "region_code",
]
CREDIT_FEATURES = [
    "loan_amount",
    "loan_term_months",
    "monthly_payment",
    "past_due_count_12m",
    "credit_history_months",
    "debt_to_income_known",
]
CLIENT_KEY_COLS = ["client_id", "branch_id", "snapshot_date"]


@dataclass(frozen=True)
class DatasetSpec:
    """Small public sample-data spec for local smoke tests.

    This generator is deliberately only a public toy data source. It does not
    create private-label files and is not the organizer's hidden generator.
    During evaluation the checker passes its own `/data` directory.
    """

    train_clients: int = 400
    test_clients: int = 160
    train_branches: int = 12
    test_branches: int = 5
    credit_train_frac: float = 0.70
    credit_test_frac: float = 0.65
    seed: int = 20261011
    random_folds: int = 5
    group_folds: int = 5


def _branch_ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(n)]


def _make_clients(rng: np.random.Generator, n: int, branches: list[str], prefix: str, include_target: bool) -> pd.DataFrame:
    branch = rng.choice(branches, size=n, replace=True)
    branch_code = np.array([int(x[1:]) for x in branch], dtype=float)
    stability = rng.normal(0, 1, n) + 0.03 * branch_code
    account_age = rng.integers(30, 3600, n)
    avg_inflow_90d = np.maximum(100, rng.lognormal(7.55 + 0.28 * stability, 0.38, n))
    avg_inflow_30d = np.maximum(50, avg_inflow_90d * rng.normal(1.0, 0.18, n))
    avg_outflow_90d = np.maximum(40, avg_inflow_90d * rng.uniform(0.45, 0.95, n))
    avg_outflow_30d = np.maximum(30, avg_outflow_90d * rng.normal(1.0, 0.18, n))
    balance_mean = (avg_inflow_90d - avg_outflow_90d) * rng.uniform(1.0, 3.0, n) + rng.normal(600, 700, n)
    balance_std = np.maximum(5, np.abs(rng.normal(0.16 * avg_inflow_30d, 180, n)))
    txn_count = np.maximum(1, rng.poisson(np.clip(avg_outflow_30d / 280, 2, 65))).astype(int)
    shares = rng.dirichlet([4.0, 2.4, 1.7], size=n)
    has_app = rng.binomial(1, 1 / (1 + np.exp(-(0.2 + 0.25 * stability))), n)
    df = pd.DataFrame({
        "client_id": [f"{prefix}{i:06d}" for i in range(n)],
        "branch_id": branch,
        "snapshot_date": "2026-01-31",
        "account_age_days": account_age,
        "avg_inflow_30d": avg_inflow_30d.round(2),
        "avg_inflow_90d": avg_inflow_90d.round(2),
        "avg_outflow_30d": avg_outflow_30d.round(2),
        "avg_outflow_90d": avg_outflow_90d.round(2),
        "balance_mean_30d": balance_mean.round(2),
        "balance_std_30d": balance_std.round(2),
        "txn_count_30d": txn_count,
        "spend_food_share": shares[:, 0].round(5),
        "spend_transport_share": shares[:, 1].round(5),
        "spend_entertainment_share": shares[:, 2].round(5),
        "has_mobile_app": has_app,
        "region_code": (branch_code.astype(int) % 5).astype(str),
    })
    if include_target:
        inflow_next = np.maximum(0, 0.55 * avg_inflow_30d + 0.35 * avg_inflow_90d + 90 * stability + rng.normal(0, 480, n))
        df["inflow_next_30d"] = inflow_next.round(2)
    for col, p in [("balance_std_30d", 0.025), ("spend_entertainment_share", 0.02)]:
        df.loc[rng.random(n) < p, col] = np.nan
    return df


def _make_credit(rng: np.random.Generator, clients: pd.DataFrame, frac: float, prefix: str, include_target: bool) -> pd.DataFrame:
    chosen = clients.sample(frac=frac, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    n = len(chosen)
    avg_income = chosen["avg_inflow_90d"].to_numpy()
    loan_amount = np.maximum(500, rng.lognormal(9.05, 0.48, n))
    term = rng.choice([6, 12, 18, 24, 36, 48], size=n)
    monthly_payment = loan_amount / term * rng.uniform(1.04, 1.20, n)
    past_due = rng.poisson(0.25, n)
    history = rng.integers(0, 96, n)
    df = pd.DataFrame({
        "application_id": [f"{prefix}{i:06d}" for i in range(n)],
        "client_id": chosen["client_id"].to_numpy(),
        "loan_amount": loan_amount.round(2),
        "loan_term_months": term,
        "monthly_payment": monthly_payment.round(2),
        "past_due_count_12m": past_due,
        "credit_history_months": history,
        "debt_to_income_known": (monthly_payment / np.maximum(200, avg_income)).round(5),
    })
    if include_target:
        # Public toy labels only for local experiments; not the private scorer.
        logits = -2.0 + 1.8 * df["debt_to_income_known"].to_numpy() + 0.35 * past_due - 0.014 * np.minimum(history, 60)
        prob = 1 / (1 + np.exp(-logits))
        df["overdue_next_60d"] = rng.binomial(1, np.clip(prob, 0.02, 0.95), n)
    return df


def _random_folds(client_ids: Iterable[str], n_splits: int, seed: int) -> pd.DataFrame:
    ids = pd.Series(list(client_ids), name="client_id").sort_values().reset_index(drop=True)
    fold = np.empty(len(ids), dtype=int)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (_, val_idx) in enumerate(kf.split(ids)):
        fold[val_idx] = fold_id
    return pd.DataFrame({"client_id": ids, "fold_id": fold})


def _group_folds(clients: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    base = clients.loc[:, ["client_id", "branch_id"]].sort_values(by=["branch_id", "client_id"]).reset_index(drop=True)
    if base["branch_id"].nunique() < n_splits:
        raise ValueError(f"Need at least {n_splits} branches")
    fold = np.empty(len(base), dtype=int)
    for fold_id, (_, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(base, groups=base["branch_id"])):
        fold[val_idx] = fold_id
    return pd.DataFrame({"client_id": base["client_id"], "fold_id": fold})


def generate_dataset(output_dir: str | Path, spec: DatasetSpec = DatasetSpec()) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    train_branches = _branch_ids("B", spec.train_branches)
    test_branches = train_branches[: max(1, spec.test_branches // 2)] + _branch_ids("N", spec.test_branches)

    clients_train = _make_clients(rng, spec.train_clients, train_branches, "C", include_target=True)
    clients_test = _make_clients(rng, spec.test_clients, test_branches, "T", include_target=False)
    credit_train = _make_credit(rng, clients_train, spec.credit_train_frac, "A", include_target=True)
    credit_test = _make_credit(rng, clients_test, spec.credit_test_frac, "Z", include_target=False)
    folds_random = _random_folds(clients_train["client_id"], spec.random_folds, spec.seed)
    folds_group = _group_folds(clients_train, spec.group_folds)

    frames = {
        "clients_train": clients_train,
        "clients_test": clients_test,
        "credit_train": credit_train,
        "credit_test": credit_test,
        "folds_random": folds_random,
        "folds_group": folds_group,
    }
    files = {key: f"{key}.csv" for key in frames}
    for key, frame in frames.items():
        frame.to_csv(out / files[key], index=False)
    meta = {
        "schema_version": 1,
        "dataset_version": f"greenbank_public_sample_v1_seed_{spec.seed}",
        "seed": spec.seed,
        "files": files,
        "rows": {k: int(len(v)) for k, v in frames.items()},
        "client_features": CLIENT_FEATURES,
        "credit_features": CREDIT_FEATURES,
        "notes": "Public sample data for local smoke tests only; private scoring data is generated separately by organizers.",
    }
    (out / "dataset_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {k: str(out / v) for k, v in files.items()}
