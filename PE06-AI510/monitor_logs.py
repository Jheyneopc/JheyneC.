from __future__ import annotations

import datetime
import random

from utils import dry_run_log, get_cloudwatch_client


def main() -> None:
    _ = get_cloudwatch_client()  # Not used for real calls; kept for structure

    dry_run_log("CloudWatch", "Fetching model inference metrics")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    requests = 150
    latency_ms = random.randint(40, 60)
    error_rate = 0.5

    print("[CloudWatch Simulation]")
    print(f"Timestamp: {timestamp}")
    print(f"Requests served: {requests}")
    print(f"Average latency: {latency_ms}ms")
    print(f"Error rate: {error_rate}%")


if __name__ == "__main__":
    main()