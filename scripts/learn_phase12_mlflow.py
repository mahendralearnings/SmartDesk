# scripts/learn_phase12_mlflow.py
from dotenv import load_dotenv
load_dotenv()

from smartdesk.mlops.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

# Simulating 5 runs with different rank values
# In real life these come from actual training
runs = [
    {
        "config": {
            "lora_r": 4,
            "lora_alpha": 8,
            "learning_rate": 3e-4,
            "frozen_layers": 10,
            "num_epochs": 3,
        },
        "metrics": {
            "f1_score":  0.81,
            "eval_loss": 0.24,
        }
    },
    {
        "config": {
            "lora_r": 8,
            "lora_alpha": 16,
            "learning_rate": 3e-4,
            "frozen_layers": 10,
            "num_epochs": 3,
        },
        "metrics": {
            "f1_score":  0.91,
            "eval_loss": 0.12,
        }
    },
    {
        "config": {
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 3e-4,
            "frozen_layers": 10,
            "num_epochs": 3,
        },
        "metrics": {
            "f1_score":  0.89,
            "eval_loss": 0.14,
        }
    },
    {
        "config": {
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 3e-4,
            "frozen_layers": 10,
            "num_epochs": 3,
        },
        "metrics": {
            "f1_score":  0.87,
            "eval_loss": 0.18,
        }
    },
    {
        "config": {
            "lora_r": 64,
            "lora_alpha": 128,
            "learning_rate": 3e-4,
            "frozen_layers": 10,
            "num_epochs": 3,
        },
        "metrics": {
            "f1_score":  0.85,
            "eval_loss": 0.22,
        }
    },
]

print("Logging 5 LoRA training runs to MLflow...")
print("=" * 45)

for run in runs:
    tracker.log_lora_run(run["config"], run["metrics"])

print("\nAll runs logged.")
print("\nNow open the MLflow dashboard:")
print("  mlflow ui")
print("  Open: http://localhost:5000")