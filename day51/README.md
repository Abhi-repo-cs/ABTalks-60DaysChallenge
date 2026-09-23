# Customer Intelligence Platform — Day 51

## Capstone Phase: Explainable AI

Day 51 adds an explainability layer to the customer churn model.

### What is included
- Explainable AI notebook
- Random Forest feature importance
- SHAP TreeExplainer analysis where supported
- Global importance visualization
- Local customer explanation
- Business interpretation report
- Saved explainable model
- Generated outputs

### Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\explainability.py
```

For SHAP support, the requirements include `shap`.

### Outputs

```text
outputs/
  feature_importance.csv
  shap_global_importance.csv        # if SHAP succeeds
  customer_001_shap_explanation.csv # if SHAP succeeds
  explainability_summary.json
  shap_fallback_reason.txt          # if SHAP fails

figures/
  global_feature_importance.png
  shap_global_importance.png        # if SHAP succeeds
  local_customer_shap.png          # if SHAP succeeds
```

### Important interpretation rule

Feature importance and SHAP explain model behavior. They do not establish causal relationships.

The included dataset is a demonstration dataset. Replace it with the real capstone dataset before making business claims.
