"""
Interactive Code Testing & Grading Script
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Usage:
    python test_code.py
"""

import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.predict import CodePredictor

SAMPLE_CODE = """def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
"""

def main():
    print("\n" + "=" * 65)
    print("AI CODE EVALUATOR - SINGLE CODE ANALYSIS DEMO")
    print("=" * 65 + "\n")

    predictor = CodePredictor()

    print("Analyzing Sample Code (Binary Search):\n")
    print(SAMPLE_CODE)
    print("-" * 65)

    result = predictor.analyze_code(SAMPLE_CODE)

    print("\n>>> PREDICTION RESULTS:")
    print(f"Time Complexity  : {result['time_complexity']['predicted_class']} (Confidence: {result['time_complexity']['confidence']}%)")
    print(f"Space Complexity : {result['space_complexity']['predicted_class']} (Confidence: {result['space_complexity']['confidence']}%)")
    print(f"Code Quality     : {result['code_quality']['overall_score']} / 100.0")
    print(f"Nesting Depth    : {result['code_quality']['max_nesting_depth']}")
    print(f"Cyclomatic Comp  : {result['code_quality']['cyclomatic_complexity']}")

    print("\n>>> RECOMMENDATIONS:")
    for rec in result["recommendations"]:
        print(f"  * {rec}")

    print("\n>>> ANALYSIS LATENCY:")
    for k, v in result["latency_breakdown_ms"].items():
        print(f"  * {k}: {v} ms")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
