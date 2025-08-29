"""
Train ranking model. Candidate generation step should run first (generate collab candidates with scores).
This script expects:
- processed events
- collab candidates with collab_score in candidates.parquet
"""
import argparse
import os
import pandas as pd
from src.models.ranker import build_features, train_ranker

def main(args):
    processed = args.data
    candidates = pd.read_parquet(os.path.join(processed, "candidates.parquet"))
    interactions = pd.read_parquet(os.path.join(processed, "events.parquet"))
    train_df = build_features(interactions, candidates)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": args.learning_rate,
        "num_leaves": 31,
        "verbose": -1
    }
    model = train_ranker(train_df, params, args.out)
    print("Saved ranker to", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--collab", required=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    args = parser.parse_args()
    main(args)
