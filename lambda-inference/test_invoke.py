"""
test_invoke.py
==============
Invokes the deployed Lambda with a real image and reports timing.
Run after deploy.ps1 to validate the deployed function.

Usage (PowerShell):
    python lambda-inference/test_invoke.py `
        --function  "fashion-inference-jackets" `
        --region    "eu-west-1" `
        --image     "E:/fashion-data/01-RAW/jackets_img/some_jacket.jpg" `
        --runs      3        # invoke N times to see cold + warm split
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import boto3


def invoke_lambda(client, function_name: str, image_path: Path) -> tuple[dict, float]:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps({"image_b64": image_b64})

    t0 = time.perf_counter()
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=payload.encode(),
    )
    round_trip_ms = (time.perf_counter() - t0) * 1000

    body = json.loads(response["Payload"].read())
    if "body" in body:
        result = json.loads(body["body"])
    else:
        result = body

    return result, round_trip_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--function", required=True)
    p.add_argument("--region",   required=True)
    p.add_argument("--image",    type=Path, required=True)
    p.add_argument("--runs",     type=int, default=3)
    args = p.parse_args()

    client = boto3.client("lambda", region_name=args.region)

    print(f"\nInvoking {args.function}  ({args.runs} runs)\n")

    for i in range(args.runs):
        result, round_trip = invoke_lambda(client, args.function, args.image)

        timing = result.get("timing", {})
        label  = "COLD" if "cold_start_ms" in timing else "WARM"
        cold   = f"  cold_start={timing.get('cold_start_ms', '-'):.0f}ms" if label == "COLD" else ""

        print(f"  Run {i+1} [{label}]")
        print(f"    predicted : {result.get('predicted_class')}  ({result.get('confidence', 0):.4f})")
        print(f"    inference : {timing.get('inference_ms', 0):.1f} ms (Lambda-side)")
        print(f"    round-trip: {round_trip:.0f} ms (total incl. network){cold}")
        if label == "COLD":
            print(f"    ★ Cold start breakdown: {timing.get('cold_start_ms', 0):.0f}ms load "
                  f"+ {timing.get('inference_ms', 0):.0f}ms inference "
                  f"= {timing.get('cold_start_ms', 0) + timing.get('inference_ms', 0):.0f}ms Lambda-side")
        print()


if __name__ == "__main__":
    main()
