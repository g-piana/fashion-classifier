# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# AWS Lambda container image — PyTorch ResNet50 inference
#
# Uses the official AWS Lambda Python base image so the Lambda runtime
# is already wired correctly (no need for custom entrypoints).
#
# PyTorch CPU-only wheel keeps the image under 2 GB.
# Full GPU torch would be ~4 GB and Lambda has no GPU anyway.
# opencv-python-headless bundles its own libs — no system deps needed.
# ─────────────────────────────────────────────────────────────────────────────

FROM public.ecr.aws/lambda/python:3.11

# Copy requirements first — Docker layer caching means pip only reruns
# when requirements.txt changes, not on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py ${LAMBDA_TASK_ROOT}/

# Lambda looks for: <module>.<function>
CMD ["handler.lambda_handler"]
