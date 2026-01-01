#!/bin/bash
# MBench Evaluation Script
# Usage: bash scripts/evaluate_mbench.sh <evaluation_path> [dimension1 dimension2 ...]

set -e

EVALUATION_PATH=${1:-"./exp/inference_results"}
shift || true
DIMENSIONS="$@"

echo "=== MBench Evaluation ==="
echo "Evaluation path: ${EVALUATION_PATH}"
echo "Dimensions: ${DIMENSIONS:-'all'}"
echo ""

if [ -z "$DIMENSIONS" ]; then
    python evaluate_mbench.py \
        --evaluation_path "$EVALUATION_PATH" \
        --output_path "./evaluation_results/"
else
    python evaluate_mbench.py \
        --evaluation_path "$EVALUATION_PATH" \
        --output_path "./evaluation_results/" \
        --dimension $DIMENSIONS
fi

echo ""
echo "=== Evaluation Complete ==="
