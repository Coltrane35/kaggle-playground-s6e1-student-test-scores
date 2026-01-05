from __future__ import annotations

import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.config import OUTPUTS_DIR, TARGET_COL, ID_COL
from src.load_data import load_train_test, load_sample_submission


def predict_and_make_submission() -> pd.DataFrame:
    train_df, test_df = load_train_test()

    model_path = OUTPUTS_DIR / "catboost_regressor.cbm"
    model = CatBoostRegressor()
    model.load_model(str(model_path))

    feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]
    cat_cols = [c for c in feature_cols if train_df[c].dtype == "object"]
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    test_pool = Pool(test_df[feature_cols], cat_features=cat_idx)
    preds = model.predict(test_pool)

    sub = load_sample_submission()
    sub[TARGET_COL] = preds

    out_path = OUTPUTS_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)

    print(f"[Saved] submission -> {out_path}")
    return sub


if __name__ == "__main__":
    predict_and_make_submission()
