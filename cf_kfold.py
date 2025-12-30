#!/usr/bin/env python3
"""
run_cv.py

独立交叉验证脚本（不修改 CF.py）。
Usage example:
  python run_cv.py --ratings ./data/ml-latest-small/ratings.csv --k 5 --min-common-user 2 --save-topk 50

要求:
  - 将此脚本和你的 CF.py 放在同一目录，或用 --cf-path 指定包含 CF.py 的目录。
  - CF.py 中应包含类 CollaborativeFiltering(train(df), predict(user, item))。

输出:
  - 每折 RMSE/MAE/coverage，并打印整体平均与标准差。
"""
import os
import sys
import argparse
import math
from collections import defaultdict
from typing import List
import numpy as np
import pandas as pd

def import_cf_module(cf_path: str):
    # 把 cf_path 加入 sys.path，优先导入该目录下的 CF.py
    if cf_path and cf_path not in sys.path:
        sys.path.insert(0, cf_path)
    try:
        from CF import CollaborativeFiltering
    except Exception as e:
        raise ImportError(f"无法导入 CF.py（路径 {cf_path}）。错误：{e}")
    return CollaborativeFiltering

def make_user_stratified_folds(df: pd.DataFrame, k: int = 5, seed: int = 42,
                              user_col: str = 'userId', item_col: str = 'movieId'):
    """
    per-user stratified folds:
      - 把 item 只出现一次的记录固定到训练集
      - 把用户只有一条记录（除去上面已固定的）固定到训练集
      - 其余用户内随机打乱后 round-robin 分配到 k 个 folds 作为验证
    返回 list of {'train_idx', 'val_idx'}
    """
    np.random.seed(seed)
    n = len(df)
    all_indices = np.arange(n)

    item_counts = df[item_col].value_counts().to_dict()

    user2indices = defaultdict(list)
    for idx, u in enumerate(df[user_col].values):
        user2indices[u].append(idx)

    folds = [[] for _ in range(k)]
    always_train = set()

    # 固定单次出现的 item 到训练集中
    for idx, item in enumerate(df[item_col].values):
        if item_counts.get(item, 0) == 1:
            always_train.add(idx)

    # per-user distribute remaining
    for u, idx_list in user2indices.items():
        candidates = [i for i in idx_list if i not in always_train]
        if len(candidates) == 0:
            continue
        if len(candidates) == 1:
            always_train.add(candidates[0])
            continue
        np.random.shuffle(candidates)
        for pos, idx in enumerate(candidates):
            fold_id = pos % k
            folds[fold_id].append(idx)

    # safety: remove always_train from folds
    for f in range(k):
        folds[f] = [i for i in folds[f] if i not in always_train]

    results = []
    all_idx_set = set(all_indices)
    for f in range(k):
        val_idx = set(folds[f])
        train_idx = list(all_idx_set - val_idx)
        results.append({
            'train_idx': sorted(train_idx),
            'val_idx': sorted(val_idx)
        })
    return results

def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def safe_predict(cf, user_id, movie_id, global_mean: float):
    """
    对 cf.predict 做保护包装：
      - 若成功且返回有限值 -> (pred, True)
      - 否则回退到 user_mean 或 global_mean -> (fallback, False)
    """
    fallback = getattr(cf, "user_mean", {}).get(user_id, global_mean)
    try:
        pred = cf.predict(user_id, movie_id)
        if pred is None or (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred))):
            return float(fallback), False
        return float(pred), True
    except Exception:
        return float(fallback), False

def run_k_fold_cv(ratings_df: pd.DataFrame, CollaborativeFilteringClass,
                  k: int = 5, min_common_user: int = 2, save_topk: int = 50,
                  seed: int = 2025, verbose: bool = True):
    folds = make_user_stratified_folds(ratings_df, k=k, seed=seed,
                                       user_col='userId', item_col='movieId')
    global_mean = float(ratings_df['rating'].mean())

    fold_results = []
    for i, f in enumerate(folds, 1):
        if verbose:
            print(f"\n--- Fold {i}/{k} ---")
            print(f"train_size={len(f['train_idx'])}, val_size={len(f['val_idx'])}")

        train_df = ratings_df.iloc[f['train_idx']].reset_index(drop=True)
        val_df = ratings_df.iloc[f['val_idx']].reset_index(drop=True)

        # 用你原来的类（不做修改）
        cf = CollaborativeFilteringClass(min_common_user=min_common_user, save_topk=save_topk)
        cf.train(train_df)

        y_true = []
        y_pred = []
        used_count = 0
        total = 0

        for row in val_df.itertuples(index=False):
            uid = row.userId
            mid = row.movieId
            true_r = float(row.rating)
            pred, used_model = safe_predict(cf, uid, mid, global_mean)
            y_true.append(true_r)
            y_pred.append(pred)
            used_count += 1 if used_model else 0
            total += 1

        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)

        fold_rmse = rmse_np(y_true, y_pred)
        fold_mae = mae_np(y_true, y_pred)
        coverage = used_count / total if total > 0 else 0.0

        fold_results.append({
            'rmse': fold_rmse,
            'mae': fold_mae,
            'coverage': coverage,
            'train_size': len(f['train_idx']),
            'val_size': len(f['val_idx'])
        })

        if verbose:
            print(f"Fold {i} RMSE={fold_rmse:.4f}, MAE={fold_mae:.4f}, coverage={coverage:.3%}")

    # summary
    rmses = [r['rmse'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]
    covs = [r['coverage'] for r in fold_results]

    summary = {
        'per_fold': fold_results,
        'avg_rmse': float(np.mean(rmses)),
        'std_rmse': float(np.std(rmses)),
        'avg_mae': float(np.mean(maes)),
        'std_mae': float(np.std(maes)),
        'avg_coverage': float(np.mean(covs))
    }

    if verbose:
        print("\n=== Summary ===")
        for idx, r in enumerate(fold_results, 1):
            print(f"Fold {idx}: RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}, coverage={r['coverage']:.3%}")
        print(f"AVG RMSE={summary['avg_rmse']:.4f} (std {summary['std_rmse']:.4f})")
        print(f"AVG MAE = {summary['avg_mae']:.4f} (std {summary['std_mae']:.4f})")
        print(f"AVG coverage = {summary['avg_coverage']:.3%}")

    return summary

def parse_args():
    p = argparse.ArgumentParser(description="Run k-fold CV for your item-based CF (without modifying CF.py).")
    p.add_argument("--ratings", type=str, required=True, help="ratings CSV path (must contain userId,movieId,rating)")
    p.add_argument("--k", type=int, default=10, help="number of folds")
    p.add_argument("--min-common-user", type=int, default=2, help="min_common_user passed to CollaborativeFiltering")
    p.add_argument("--save-topk", type=int, default=50, help="save_topk passed to CollaborativeFiltering (useful to limit memory)")
    p.add_argument("--seed", type=int, default=2025, help="random seed")
    p.add_argument("--cf-path", type=str, default=".", help="path to directory containing CF.py (default: current dir)")
    p.add_argument("--no-verbose", action="store_true", help="suppress per-fold printing")
    return p.parse_args()

def main():
    args = parse_args()
    CollaborativeFilteringClass = import_cf_module(args.cf_path)
    df = pd.read_csv(args.ratings)
    required_cols = {'userId', 'movieId', 'rating'}
    if not required_cols.issubset(set(df.columns)):
        raise RuntimeError(f"ratings 文件必须包含列: {required_cols}")
    run_k_fold_cv(df, CollaborativeFilteringClass, k=args.k,
                  min_common_user=args.min_common_user, save_topk=args.save_topk,
                  seed=args.seed, verbose=not args.no_verbose)

if __name__ == "__main__":
    main()
