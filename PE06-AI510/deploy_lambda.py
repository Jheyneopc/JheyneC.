from __future__ import annotations

from utils import dry_run_log, get_sagemaker_client


def main() -> None:
    _ = get_sagemaker_client()  # Not used for real calls; kept for structure

    endpoint_name = "iris-endpoint-demo"
    dry_run_log("SageMaker", f"Deploying model to endpoint: {endpoint_name}")
    print("Simulated deployment complete.")


if __name__ == "__main__":
    main()