# Predicting Student Test Scores  
### Kaggle Playground Series — Season 6, Episode 1

End-to-end machine learning pipeline (pure Python scripts, no notebooks) built for the Kaggle Playground Series S6E1 competition.  
The task is to predict students' **exam scores** based on demographic, study-related, and lifestyle features.

---

## 🧠 Problem Description
- **Type:** Supervised Learning — Regression  
- **Target variable:** `exam_score`  
- **Evaluation metric:** RMSE (Root Mean Squared Error)

---

## 🏗️ Solution Overview
The project follows a clean, reproducible workflow implemented entirely in `.py` files:

1. Data loading and validation  
2. Exploratory Data Analysis (EDA) with automatic report generation  
3. Feature selection and preparation  
4. Model training using **CatBoostRegressor**  
5. 5-fold Cross-Validation (OOF evaluation)  
6. Final model training on full dataset  
7. Prediction and Kaggle submission generation  

**CatBoost** was selected due to its strong performance on tabular data and native handling of categorical features.

---

## 📊 Results
- **OOF RMSE (5-fold CV):** 8.7470  
- **Public Kaggle Score:** **8.71656**  

The close alignment between OOF and public score indicates good generalization and stable model performance.

---

## 🧰 Tech Stack
- Python 3  
- pandas  
- numpy  
- scikit-learn  
- CatBoost  

---

## 📁 Project Structure
```
playground-series-s6e1/
├── data/                 # Kaggle data (not tracked in Git)
├── outputs/              # Model artifacts & reports
│   ├── metrics.txt
│   ├── submission.csv
│   └── eda_report.txt
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── eda.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run (Windows / PowerShell)

### 1️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download competition data (optional)
```bash
kaggle competitions download -c playground-series-s6e1 -p data
```

### 4️⃣ Run EDA
```bash
python -m src.eda
```

### 5️⃣ Train model (Cross-Validation + final model)
```bash
python -m src.train
```

### 6️⃣ Generate submission
```bash
python -m src.predict
```

All outputs are saved in the `outputs/` directory.

---

## 📌 Notes
- The pipeline is designed to be **modular and reusable** for other Kaggle Playground competitions.
- Training on the full dataset is computationally expensive but provides a strong and reliable baseline.
- No Jupyter notebooks were used — the project is fully script-based.

---

## 👤 Author
**Grzegorz Rączka**  
Machine Learning / Data Science  

---

## 🔗 Kaggle Competition
https://www.kaggle.com/competitions/playground-series-s6e1
