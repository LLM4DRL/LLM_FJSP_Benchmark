import os
import json
from datetime import datetime

def save_output_and_metrics(output, metrics, deployment_type, model_id):
    """Save raw LLM output and corresponding metrics to the results folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("results", deployment_type, model_id)
    os.makedirs(base_dir, exist_ok=True)
    # Save output text
    out_path = os.path.join(base_dir, f"run_{timestamp}_output.txt")
    with open(out_path, "w") as f:
        f.write(output)
    # Save metrics JSON
    metrics_path = os.path.join(base_dir, f"run_{timestamp}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2) 