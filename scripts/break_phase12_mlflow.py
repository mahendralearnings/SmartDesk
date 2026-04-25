# scripts/break_phase12_mlflow.py
from dotenv import load_dotenv
load_dotenv()

import mlflow
import time

# Clean up any leftover active run first
mlflow.end_run()

# Update tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

SEP = "=" * 45

# BREAK 1
print(SEP)
print("BREAK 1: Log without starting a run")
print(SEP)
try:
    mlflow.log_metric("f1_score", 0.91)
except Exception as e:
    print(f"Error caught: {type(e).__name__}")
    print("Fix: always use 'with mlflow.start_run():'")

# BREAK 2
print(f"\n{SEP}")
print("BREAK 2: Wrong data type for metric")
print(SEP)
with mlflow.start_run(run_name=f"break-type-{int(time.time())}"):
    try:
        mlflow.log_metric("f1_score", "very good")
    except Exception as e:
        print(f"Error caught: {type(e).__name__}")
        print("Fix: metrics must be int or float")

# BREAK 3
print(f"\n{SEP}")
print("BREAK 3: Duplicate run names")
print(SEP)
mlflow.set_experiment("smartdesk-ner-lora")
for i in range(3):
    with mlflow.start_run(run_name="lora-r8-duplicate"):
        mlflow.log_param("lora_r", 8)
        mlflow.log_metric("f1_score", 0.91 + i * 0.01)
        print(f"  Run {i+1} logged")

print("\nCheck dashboard — 3 rows with same name")
print("Fix: use f'lora-r8-{int(time.time())}' for unique names")
print(f"\n{SEP}")
print("All 3 breaks complete")
print(SEP)