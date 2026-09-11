# Day 40 — A/B Testing Analytics

A practical A/B testing project that compares control and experiment groups, calculates statistical significance, visualizes experiment outcomes, and translates the result into a business decision.

## Project Structure
- `data/ab_test_data.csv` — simulated user-level A/B test dataset
- `day40_ab_testing.ipynb` — analysis notebook
- `outputs/experiment_analysis_report.md` — experiment analysis report
- `requirements.txt` — Python dependencies

## Metrics
- Conversion rate
- Absolute conversion lift
- Relative conversion lift
- Revenue per user
- Average session duration
- z-statistic
- p-value

## Statistical Test
Two-proportion z-test with α = 0.05.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook day40_ab_testing.ipynb
```

## Business Impact
A/B testing enables product and marketing teams to make evidence-based decisions instead of relying only on intuition. The analysis also highlights why statistical significance should be combined with practical business impact and guardrail metrics.

## Skills Demonstrated
Python, Pandas, NumPy, SciPy, Matplotlib, hypothesis testing, experimentation analytics, KPI analysis, and business decision-making.
