# Reliability Test Checklist
- Reject age below 18
- Reject negative numeric fields
- Accept valid prediction payload
- `/health` returns 200
- `/ready` reports model readiness
- Empty dashboard results render safely
- Invalid requests return validation errors
- Prediction failures do not expose stack traces
