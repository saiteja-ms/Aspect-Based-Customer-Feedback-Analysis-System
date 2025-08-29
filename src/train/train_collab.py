"""
Train collaborative component (matrix factorization) and log with MLflow.
"""
import argparse
import os
import mlflow
import pandas as pd
from src.models.collab_matrix import SimpleMF
from src.mlflow_tracking.mlflow_utils import start_mlflow_run

def main(args):
    df = pd.read_parquet(os.path.join(args.data, "interactions.parquet"))
    mf = SimpleMF(n_factors=args.n_factors)
    with start_mlflow_run("train_collab") as run:
        mf.fit(df)
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        mf.save(os.path.join(out_dir, "collab.pkl"))
        mlflow.log_artifact(os.path.join(out_dir, "collab.pkl"))
        mlflow.log_param("n_factors", args.n_factors)
        print("Saved collab model to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_factors", type=int, default=64)
    args = parser.parse_args()
    main(args)
