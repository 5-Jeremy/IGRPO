"""Build the validation subset used during training from the Search-R1 test split.

The full test split is 51,713 examples across 7 datasets. Validating on all of
it every `trainer.test_freq` steps costs about as much rollout compute as
training itself, so training runs point `data.val_files` here and keep the full
test split for final evaluation.

Default is HotpotQA only, which is the multi-hop half of the training mixture
(the other half is Natural Questions). `_validate` reports
`val/{data_source}/test_score`, so a single-source subset gives one clean curve.
"""

import argparse
import os

import pandas as pd

DEFAULT_DIR = "/scratch/project/prj-02-llm-reasoning-shakkottai/debajoy/IGRPO/searchR1_processed_direct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=DEFAULT_DIR, help="Directory holding test.parquet.")
    parser.add_argument("--sources", nargs="+", default=["hotpotqa"], help="data_source values to keep.")
    parser.add_argument("--n", type=int, default=1024, help="Examples per source. 0 keeps all of them.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_name", default="val_subset.parquet")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

    available = set(df["data_source"].unique())
    missing = [s for s in args.sources if s not in available]
    if missing:
        raise SystemExit(f"data_source not present in test.parquet: {missing}\navailable: {sorted(available)}")

    subsets = []
    for source in args.sources:
        group = df[df["data_source"] == source]
        n = len(group) if args.n == 0 else min(args.n, len(group))
        if n < args.n:
            print(f"  {source}: only {len(group)} available, taking all {n}")
        subsets.append(group.sample(n=n, random_state=args.seed))

    subset = pd.concat(subsets).sort_index()
    out_path = os.path.join(data_dir, args.out_name)
    subset.to_parquet(out_path, index=False)

    print(f"\nWrote {len(subset)} rows to {out_path}")
    print(subset["data_source"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
