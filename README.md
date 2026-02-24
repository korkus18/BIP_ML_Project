# BIP Machine Learning Project — Bank Marketing

**Blended Intensive Program (BIP)**  
*Machine Learning: Mathematical aspects, techniques, and applications — 2nd edition*

Classification project using the **Bank Marketing** dataset to predict whether a client will subscribe to a term deposit.

**Authors:** Jakub Korous, Eliška Zajacová  
**Date:** 2026

---

## Overview

This project implements a full machine learning pipeline:

1. **Exploratory Data Analysis (EDA)** — structure, distributions, correlations, target vs features  
2. **Data Preprocessing** — encoding, train/test split, scaling, class imbalance handling  
3. **Model Training & Evaluation** — Logistic Regression, Decision Tree, Random Forest (default and tuned)  
4. **Report Generation** — academic PDF report with methodology, math, and results

The goal is to predict the binary target `y` (yes/no) for term deposit subscription to support bank marketing campaigns.

---

## Dataset

- **Name:** Bank Marketing Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
- **File used:** `bank.csv` (semicolon-separated)

Place the dataset so that the main script can read it at:

```text
../bank+marketing/bank/bank.csv
```

(i.e. one level above the project folder: `BIP_ML_Project/` → `../bank+marketing/bank/bank.csv`).

If your path is different, change the path in `main.py` (around line 54):

```python
df = pd.read_csv('../bank+marketing/bank/bank.csv', sep=';')
```

---

## Requirements

- Python 3.8+  
- Dependencies in `requirements.txt`:

```text
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```

---

## Installation

1. Clone or download the project and go to its directory:

   ```bash
   cd BIP_ML_Project
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   # or: venv\Scripts\activate   # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the Bank Marketing data is at `../bank+marketing/bank/bank.csv` (or update the path in `main.py`).

---

## Usage

### 1. Run the main pipeline (EDA, preprocessing, training, comparison)

```bash
python main.py
```

This will:

- Load the dataset and run EDA  
- Save figures in `figures/`  
- Preprocess data and save `preprocessor.pkl`  
- Train 6 models (default + tuned for each algorithm)  
- Save metrics to `model_comparison.csv` and `summary_stats.json`  
- Save models to `models.pkl`

### 2. Generate the PDF report

After `main.py` has been run at least once (so `summary_stats.json` and `model_comparison.csv` exist):

```bash
python generate_report.py
```

This produces **`Report.pdf`** with:

- Title, table of contents  
- Introduction, dataset description, EDA, preprocessing  
- Methodology and mathematical foundations  
- Bias–variance tradeoff  
- Model results table, analysis, comparison, conclusion  

---

## Project Structure

```text
BIP_ML_Project/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── main.py                # EDA, preprocessing, training, comparison
├── generate_report.py     # PDF report generation
├── figures/               # Generated plots (created by main.py)
├── preprocessor.pkl       # Encoder + scaler (after main.py)
├── models.pkl             # Trained models (after main.py)
├── model_comparison.csv   # Metrics table (after main.py)
├── summary_stats.json     # Summary for report (after main.py)
└── Report.pdf             # Final report (after generate_report.py)
```

---

## Models

| Model              | Default | Tuned (main hyperparameters) |
|--------------------|--------|-----------------------------|
| Logistic Regression| ✓      | `class_weight='balanced'`, `C=0.1`, `solver='liblinear'` |
| Decision Tree      | ✓      | `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`, `class_weight='balanced'` |
| Random Forest      | ✓      | `n_estimators=200`, `max_depth=15`, `min_samples_split=10`, `min_samples_leaf=5`, `class_weight='balanced'` |

Evaluation metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and confusion matrices.  
F1 and recall are emphasized because of class imbalance (~88% “no”, ~12% “yes”).

---

## Outputs

- **`figures/`** — EDA and model plots (e.g. target distribution, correlations, confusion matrices, feature importance, model comparison).  
- **`model_comparison.csv`** — Per-model accuracy, precision, recall, F1.  
- **`summary_stats.json`** — Dataset and model summary used by the report.  
- **`Report.pdf`** — Full academic report (run `generate_report.py` after `main.py`).

---

## License

For academic use within the BIP program. Dataset usage should comply with the UCI Machine Learning Repository terms.
