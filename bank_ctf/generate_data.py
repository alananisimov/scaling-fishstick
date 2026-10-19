from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .data import DatasetSpec, generate_dataset


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Green Bank Pipeline CTF synthetic data")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=DatasetSpec.seed)
    p.add_argument("--train-clients", type=int, default=DatasetSpec.train_clients)
    p.add_argument("--test-clients", type=int, default=DatasetSpec.test_clients)
    p.add_argument("--train-branches", type=int, default=DatasetSpec.train_branches)
    p.add_argument("--test-branches", type=int, default=DatasetSpec.test_branches)
    p.add_argument("--include-private-labels", action="store_true", help="keep hidden test-label files for checker/scorer use")
    args = p.parse_args(argv)
    spec = DatasetSpec(
        seed=args.seed,
        train_clients=args.train_clients,
        test_clients=args.test_clients,
        train_branches=args.train_branches,
        test_branches=args.test_branches,
    )
    generate_dataset(args.output_dir, spec)
    if not args.include_private_labels:
        out = Path(args.output_dir)
        (out / "clients_test_private.csv").unlink(missing_ok=True)
        (out / "credit_test_private.csv").unlink(missing_ok=True)
    print(f"Generated Green Bank dataset in {args.output_dir}: {asdict(spec)}")


if __name__ == "__main__":
    main()
