# Customer Intelligence Platform — Day 50

## Capstone Phase: Model Optimization

Day 50 improves the Day 49 baseline by comparing multiple algorithms, tuning hyperparameters, measuring generalization, and documenting engineering tradeoffs.

### Models
- Logistic Regression
- Random Forest
- Gradient Boosting

### Optimization
- Stratified 5-fold cross-validation
- GridSearchCV
- ROC-AUC-driven model selection
- Train/test generalization analysis

### Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\optimize_models.py
```

### Outputs

```text
outputs/
  model_comparison.csv
  optimization_summary.json

figures/
  model_comparison_roc_auc.png
  generalization_gap.png

models/
  optimized_model.joblib
```

Open the notebook:

`notebooks/day50_model_optimization.ipynb`

### Important
The included dataset is a demonstration dataset. Do not interpret its performance as production evidence. Replace it with the actual capstone dataset before drawing business conclusions.
