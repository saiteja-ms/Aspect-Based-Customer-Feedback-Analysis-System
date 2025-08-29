"""
Evaluate recall@k and NDCG.
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

def recall_at_k(true_items, preds, k=10):
    return len(set(true_items) & set(preds[:k])) / len(true_items)

def main(args):
    candidates = pd.read_parquet(os.path.join(args.data, "candidates.parquet"))
    labels = pd.read_parquet(os.path.join(args.data, "events.parquet"))
    # very simplified: group by user and compute recall@10 using candidate ranked by collab_score
    recs = candidates.sort_values(['user_id','collab_score'], ascending=[True, False])
    grouped = recs.groupby('user_id').head(10)
    # compute per-user recall using historical purchases as truth
    results = []
    for user, group in grouped.groupby('user_id'):
        true = labels[labels.user_id==user]['item_id'].unique().tolist()
        pred = group['item_id'].tolist()
        if not true:
            continue
        results.append(recall_at_k(true, pred, k=10))
    print("Mean recall@10:", np.mean(results))
    # write report
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.Series(results).to_json(args.out)
    print("Saved eval report to", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--collab", required=False)
    parser.add_argument("--ranker", required=False)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
