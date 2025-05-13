# Results Directory
This directory stores outputs and performance metrics from tests.

Structure:
```
results/<deployment_type>/<model_id>/
```
Each run generates:
- `run_<timestamp>_output.txt`: Raw LLM output
- `run_<timestamp>_metrics.json`: Performance metrics

A summary CSV may be maintained at `results/summary_report.csv` with aggregated metrics. 