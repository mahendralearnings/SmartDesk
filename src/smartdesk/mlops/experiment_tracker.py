# src/smartdesk/mlops/experiment_tracker.py
import mlflow
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")


class ExperimentTracker:

    @staticmethod
    def log_lora_run(config: dict, metrics: dict, artifact_path: str = None):
        """
        Log one complete LoRA training run.
        config  = parameters you used
        metrics = scores you got
        """
        mlflow.set_experiment("smartdesk-ner-lora")

        with mlflow.start_run(
            run_name=f"lora-r{config['lora_r']}-alpha{config['lora_alpha']}"
        ):
            # Log every parameter
            mlflow.log_params(config)

            # Log every metric
            mlflow.log_metrics(metrics)

            # Log artifact file if provided
            if artifact_path and os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)

            print(f"Run logged: lora-r{config['lora_r']}")
            print(f"  F1 score: {metrics.get('f1_score', 'N/A')}")