from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import OUTPUTS_DIR, TARGET_COL
from src.load_data import load_train_test


def basic_eda(train: pd.DataFrame, test: pd.DataFrame) -> str:
    lines: list[str] = []

    lines.append("=== SHAPES ===")
    lines.append(f"train: {train.shape}")
    lines.append(f"test : {test.shape}\n")

    lines.append("=== COLUMNS (train) ===")
    lines.append(", ".join(train.columns.tolist()) + "\n")

    # Target stats
    y = train[TARGET_COL]
    lines.append("=== TARGET (train) ===")
    lines.append(y.describe().to_string())
    lines.append("")

    # Missing values
    lines.append("=== MISSING VALUES (train) ===")
    miss_train = train.isna().mean().sort_values(ascending=False)
    miss_train = miss_train[miss_train > 0]
    lines.append(miss_train.to_string() if len(miss_train) else "No missing values.\n")

    lines.append("=== MISSING VALUES (test) ===")
    miss_test = test.isna().mean().sort_values(ascending=False)
    miss_test = miss_test[miss_test > 0]
    lines.append(miss_test.to_string() if len(miss_test) else "No missing values.\n")

    # Dtypes
    lines.append("=== DTYPES (train) ===")
    lines.append(train.dtypes.to_string())
    lines.append("")

    return "\n".join(lines)


def save_text_report(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    train_df, test_df = load_train_test()
    report = basic_eda(train_df, test_df)

    out_file = OUTPUTS_DIR / "eda_report.txt"
    save_text_report(report, out_file)

    print(report[:1500])
    print(f"\n[Saved] {out_file}")
