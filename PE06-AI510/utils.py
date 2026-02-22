from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import boto3


def dry_run_log(service: str, message: str) -> None:
    """Print a standardized simulation log line."""
    print(f"[SIMULATION] {service}: {message}")


def get_s3_client(region_name: str = "us-west-2"):
    """Return a boto3 S3 client (used only for demonstrating realistic code structure)."""
    return boto3.client("s3", region_name=region_name)


def get_sagemaker_client(region_name: str = "us-west-2"):
    """Return a boto3 SageMaker client (used only for demonstrating realistic code structure)."""
    return boto3.client("sagemaker", region_name=region_name)


def get_cloudwatch_client(region_name: str = "us-west-2"):
    """Return a boto3 CloudWatch client (used only for demonstrating realistic code structure)."""
    return boto3.client("cloudwatch", region_name=region_name)