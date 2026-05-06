from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import Project42Config, ensure_dirs
from .io_helpers import import_pandas


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (ss_res / ss_tot)


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted.")
        return (x - self.mean_) / self.std_


class OLSRegressor:
    def __init__(self):
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x1 = np.column_stack([np.ones(len(x)), x])
        self.coef_ = np.linalg.pinv(x1) @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted.")
        x1 = np.column_stack([np.ones(len(x)), x])
        return x1 @ self.coef_


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x1 = np.column_stack([np.ones(len(x)), x])
        n_features = x1.shape[1]
        reg = np.eye(n_features) * self.alpha
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(x1.T @ x1 + reg) @ (x1.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted.")
        x1 = np.column_stack([np.ones(len(x)), x])
        return x1 @ self.coef_


class KNNRegressor:
    def __init__(self, k: int = 7):
        self.k = k
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x_train = x
        self.y_train = y
        self.k = max(1, min(self.k, len(x)))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Model not fitted.")
        preds = np.zeros(len(x), dtype=float)
        for i, row in enumerate(x):
            d2 = np.sum((self.x_train - row) ** 2, axis=1)
            idx = np.argpartition(d2, self.k - 1)[: self.k]
            d = np.sqrt(np.maximum(d2[idx], 1e-12))
            w = 1.0 / d
            preds[i] = float(np.sum(w * self.y_train[idx]) / np.sum(w))
        return preds


class BasicMLPRegressor:
    def __init__(
        self,
        hidden_dim: int = 16,
        lr: float = 0.02,
        epochs: int = 120,
        batch_size: int = 32,
        l2: float = 1e-4,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.w1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.b2: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n, d = x.shape
        h = self.hidden_dim
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, h))
        self.b1 = np.zeros(h, dtype=float)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / h), size=(h, 1))
        self.b2 = 0.0

        y = y.reshape(-1, 1)
        bs = max(8, min(self.batch_size, n))

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            x_s = x[idx]
            y_s = y[idx]
            for start in range(0, n, bs):
                xb = x_s[start : start + bs]
                yb = y_s[start : start + bs]
                m = xb.shape[0]

                z1 = xb @ self.w1 + self.b1
                a1 = np.tanh(z1)
                y_hat = a1 @ self.w2 + self.b2
                err = y_hat - yb

                d_y = (2.0 / m) * err
                g_w2 = a1.T @ d_y + self.l2 * self.w2
                g_b2 = float(np.sum(d_y))
                d_a1 = d_y @ self.w2.T
                d_z1 = d_a1 * (1.0 - np.tanh(z1) ** 2)
                g_w1 = xb.T @ d_z1 + self.l2 * self.w1
                g_b1 = np.sum(d_z1, axis=0)

                self.w2 -= self.lr * g_w2
                self.b2 -= self.lr * g_b2
                self.w1 -= self.lr * g_w1
                self.b1 -= self.lr * g_b1
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w1 is None or self.b1 is None or self.w2 is None:
            raise RuntimeError("Model not fitted.")
        a1 = np.tanh(x @ self.w1 + self.b1)
        y_hat = a1 @ self.w2 + self.b2
        return y_hat.reshape(-1)


def prepare_features(df):
    df = df.copy()
    for col in ["Mn_wt", "C_wt", "Al_wt", "Si_wt", "T_anneal_C", "Time_sec", "RA_fraction"]:
        df[col] = df[col].astype(float)
    for col in ["Al_wt", "Si_wt", "Cu_wt", "Ni_wt", "Cr_wt", "Mo_wt", "Nb_wt", "V_wt"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0).astype(float)

    method = df["Measurement_Method"].fillna("").astype(str).str.lower()
    df["is_xrd"] = method.str.contains("xrd").astype(float)
    df["is_ebsd"] = method.str.contains("ebsd").astype(float)
    df["log_time"] = np.log1p(np.clip(df["Time_sec"].values, a_min=0.0, a_max=None))
    df["mn_c_interaction"] = df["Mn_wt"] * df["C_wt"]
    ac1_proxy = 723.0 - 10.7 * df["Mn_wt"] + 29.1 * df["Si_wt"] + 20.0 * df["Al_wt"] - 35.0 * df["C_wt"]
    df["temp_minus_ac1_proxy"] = df["T_anneal_C"] - ac1_proxy
    return df


def run_loso(df, seed: int = 42):
    pd = import_pandas(Project42Config().root)
    feature_cols = [
        "Mn_wt",
        "C_wt",
        "Al_wt",
        "Si_wt",
        "T_anneal_C",
        "log_time",
        "mn_c_interaction",
        "temp_minus_ac1_proxy",
        "is_xrd",
        "is_ebsd",
    ]
    target_col = "RA_fraction"

    studies = sorted(df["Study"].astype(str).unique().tolist())
    rows: List[Dict[str, float]] = []
    fold = 0
    for study in studies:
        test_df = df[df["Study"].astype(str) == study].copy()
        train_df = df[df["Study"].astype(str) != study].copy()
        if len(test_df) == 0 or len(train_df) < 12:
            continue

        fold += 1
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)

        scaler = StandardScaler().fit(x_train)
        xtr = scaler.transform(x_train)
        xte = scaler.transform(x_test)

        models = {
            "ols": OLSRegressor().fit(xtr, y_train),
            "ridge": RidgeRegressor(alpha=0.7).fit(xtr, y_train),
            "knn": KNNRegressor(k=max(3, int(np.sqrt(len(train_df))))).fit(xtr, y_train),
            "basic_nn": BasicMLPRegressor(seed=seed + fold).fit(xtr, y_train),
        }

        preds = {name: np.clip(model.predict(xte), 0.0, 1.0) for name, model in models.items()}
        for i in range(len(test_df)):
            row = {
                "Study": str(test_df["Study"].iloc[i]),
                "Year": float(test_df["Year"].iloc[i]) if "Year" in test_df.columns else math.nan,
                "y_true": float(y_test[i]),
            }
            for name, pred in preds.items():
                row[f"pred_{name}"] = float(pred[i])
            rows.append(row)

    if not rows:
        raise RuntimeError("No LOSO folds produced. Check dataset size and Study labels.")

    pred_df = pd.DataFrame(rows)
    metrics = []
    model_names = ["ols", "ridge", "knn", "basic_nn"]
    inv_rmse = {}
    for name in model_names:
        y_t = pred_df["y_true"].to_numpy(dtype=float)
        y_p = pred_df[f"pred_{name}"].to_numpy(dtype=float)
        cur_rmse = rmse(y_t, y_p)
        cur_mae = mae(y_t, y_p)
        cur_r2 = r2_score(y_t, y_p)
        metrics.append({"model": name, "rmse": cur_rmse, "mae": cur_mae, "r2": cur_r2, "n": int(len(pred_df))})
        inv_rmse[name] = 1.0 / max(cur_rmse, 1e-9)

    denom = sum(inv_rmse.values())
    weights = {k: v / denom for k, v in inv_rmse.items()}
    fused = np.zeros(len(pred_df), dtype=float)
    for name in model_names:
        fused += weights[name] * pred_df[f"pred_{name}"].to_numpy(dtype=float)
    fused = np.clip(fused, 0.0, 1.0)
    pred_df["pred_fusion"] = fused
    metrics.append(
        {
            "model": "fusion",
            "rmse": rmse(pred_df["y_true"].to_numpy(dtype=float), fused),
            "mae": mae(pred_df["y_true"].to_numpy(dtype=float), fused),
            "r2": r2_score(pred_df["y_true"].to_numpy(dtype=float), fused),
            "n": int(len(pred_df)),
        }
    )
    return pred_df, metrics, weights


def run_training(dataset_path: Path, out_prefix: str, seed: int):
    cfg = Project42Config()
    ensure_dirs(cfg)
    pd = import_pandas(cfg.root)

    df = pd.read_csv(dataset_path)
    needed = ["Study", "Mn_wt", "C_wt", "T_anneal_C", "Time_sec", "RA_fraction"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=needed).copy()
    df = df[(df["RA_fraction"] >= 0.0) & (df["RA_fraction"] <= 1.0)].copy()
    df = prepare_features(df)

    pred_df, metrics, weights = run_loso(df, seed=seed)
    metrics_df = pd.DataFrame(metrics)

    pred_path = cfg.outputs_dir / f"{out_prefix}_loso_predictions.csv"
    metrics_path = cfg.outputs_dir / f"{out_prefix}_metrics.csv"
    weights_path = cfg.outputs_dir / f"{out_prefix}_fusion_weights.json"
    pred_df.to_csv(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    with weights_path.open("w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    summary = {
        "dataset": str(dataset_path),
        "rows_used": int(len(df)),
        "studies_used": int(df["Study"].nunique()),
        "predictions_file": str(pred_path),
        "metrics_file": str(metrics_path),
        "weights_file": str(weights_path),
        "metrics": metrics,
    }
    summary_path = cfg.outputs_dir / f"{out_prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train simple ML + basic NN + fusion baselines with LOSO.")
    parser.add_argument("--dataset", type=str, default=str(Project42Config().processed_dir / "ml_ready_real_pool.csv"))
    parser.add_argument("--out-prefix", type=str, default="real_pool")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_training(Path(args.dataset), out_prefix=args.out_prefix, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
