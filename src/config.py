from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ID_COL = "id"
TARGET_COL = "exam_score"

RANDOM_STATE = 42
N_SPLITS = 5
