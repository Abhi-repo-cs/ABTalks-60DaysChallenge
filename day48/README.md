# Customer Intelligence Platform — Day 48

## Capstone Phase
**Day 48: Data Preprocessing Pipeline**

This repository contains a production-style preprocessing workflow for the Customer Intelligence Platform capstone.

## Deliverables

- Preprocessing pipeline notebook
- Reusable Python preprocessing module
- Sample raw and processed data structure
- Pipeline architecture report
- Requirements file
- Git configuration

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Open:

`notebooks/day48_preprocessing_pipeline.ipynb`

## Pipeline

Raw Data → Validation → Cleaning → Feature Engineering → Imputation → Encoding → Scaling → Processed Features

## Key Engineering Principle

Fit transformations on the training set and only transform the test set. This helps prevent data leakage.

## Project Structure

```text
data/
  raw/
  processed/
notebooks/
src/preprocessing/
reports/
models/
```

## Note

The included CSV is a small demonstration dataset so the repository is runnable. Replace it with the selected capstone dataset and update the target/features for the final project.
