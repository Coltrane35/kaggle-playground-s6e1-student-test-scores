from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor, Pool

from src.config import OUTPUTS_DIR, TARGET_COL, ID_COL, RANDOM_STATE, N_SPLITS
from src.load_data import load_train_test


def _feature_cols(train_df: pd.DataFrame) -> list[str]:
    return [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]


def _cat_cols(train_df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if train_df[c].dtype == "object"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_catboost_cv(train_df: pd.DataFrame) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = _feature_cols(train_df)
    cat_cols = _cat_cols(train_df, feature_cols)
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    X = train_df[feature_cols]
    y = train_df[TARGET_COL].astype(float).values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros(len(train_df), dtype=float)
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        val_pool = Pool(X_va, y_va, cat_features=cat_idx)

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=6000,
            learning_rate=0.05,
            depth=8,
            random_seed=RANDOM_STATE,
            verbose=300,
            allow_writing_files=False,
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=400,
        )

        va_pred = model.predict(val_pool)
        oof[va_idx] = va_pred

        fold_rmse = rmse(y_va, va_pred)
        fold_scores.append(fold_rmse)
        print(f"[Fold {fold}] RMSE: {fold_rmse:.5f} | best_iter: {model.get_best_iteration()}")

    oof_rmse = rmse(y, oof)
    mean_rmse = float(np.mean(fold_scores))
    std_rmse = float(np.std(fold_scores))

    print(f"\n[OOF] RMSE: {oof_rmse:.5f}")
    print(f"[Folds] mean: {mean_rmse:.5f} ± {std_rmse:.5f}")

    # Fit final model on full data
    full_pool = Pool(X, y, cat_features=cat_idx)

    final_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        random_seed=RANDOM_STATE,
        verbose=300,
        allow_writing_files=False,
    )
    final_model.fit(full_pool)

    model_path = OUTPUTS_DIR / "catboost_regressor.cbm"
    final_model.save_model(str(model_path))

    metrics = {
        "oof_rmse": float(oof_rmse),
        "mean_fold_rmse": mean_rmse,
        "std_fold_rmse": std_rmse,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "model_path": str(model_path),
    }

    (OUTPUTS_DIR / "metrics.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in metrics.items()]),
        encoding="utf-8",
    )

    print(f"\n[Saved] model -> {model_path}")
    print(f"[Saved] metrics -> {OUTPUTS_DIR / 'metrics.txt'}")

    return metrics


if __name__ == "__main__":
    train_df, _ = load_train_test()
    train_catboost_cv(train_df)
