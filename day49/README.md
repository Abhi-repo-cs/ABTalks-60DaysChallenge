# Customer Intelligence Platform — Day 49

## Capstone Phase: Baseline System

Day 49 integrates preprocessing, feature engineering, prediction, and evaluation into the first working capstone prototype.

### Baseline task
Customer churn prediction using Logistic Regression.

### Run the prototype

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt

python src/train_baseline.py
```

Outputs will be generated in:

```text
outputs/
├── baseline_metrics.json
├── prediction_outputs.csv
└── confusion_matrix.csv

models/
└── baseline_churn_pipeline.joblib
```

### Notebook

Open:

`notebooks/day49_baseline_system.ipynb`

### Architecture

Raw Data → Preprocessing → Feature Engineering → Logistic Regression → Predictions → Evaluation

### Important
The included dataset is a synthetic/small demonstration dataset intended to make the prototype runnable. Do not interpret its metrics as evidence of production model performance.
