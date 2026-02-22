from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from utils import dry_run_log, get_s3_client


BUCKET_NAME = "mlops-demo-bucket"
MODEL_BASENAME = "iris_model.pkl"
REGISTRY_LOG = "model_registry.log"


def now_timestamp() -> str:
    """Return timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_registry_line(timestamp: str, model_filename: str, accuracy: float) -> None:
    """Append a single metadata line to the local registry log."""
    line = f"{timestamp}, model={model_filename}, accuracy={accuracy:.2f}\n"
    with open(REGISTRY_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    # Step 1: Train model (Iris dataset)
    iris = load_iris()
    X, y = iris.data, iris.target

    # Use fixed random_state for deterministic behavior
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Compute accuracy on the full dataset to match the expected output style
    accuracy = float(model.score(X, y))
    print("Model trained successfully.")

    # Step 2: Create a timestamped folder and save model locally
    timestamp = now_timestamp()
    local_dir = Path("model") / timestamp
    ensure_dir(local_dir)

    local_model_path = local_dir / MODEL_BASENAME
    joblib.dump(model, local_model_path)
    print(f"Model saved to {local_model_path}")

    # Step 3: Simulate S3 upload path using the timestamp folder
    _ = get_s3_client()  # Not used for real upload; kept for realistic structure
    s3_key = f"models/{timestamp}/{MODEL_BASENAME}"
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

    dry_run_log("S3", f"Uploading model {local_model_path} to {s3_uri}")
    print(f"Simulated upload path: {s3_uri}")

    # Step 4: Log model metadata locally (registry)
    append_registry_line(timestamp=timestamp, model_filename=MODEL_BASENAME, accuracy=accuracy)
    dry_run_log("LOG", f"Model metadata recorded at {timestamp}")

    # Summary
    print("\nSummary:")
    print("  Model trained successfully.")
    print(f"  Simulated upload path: {s3_uri}")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Log written to: {REGISTRY_LOG}")


if __name__ == "__main__":
    main()