import subprocess
import sys
import re
from pathlib import Path

import pandas as pd


def run_pytest(test_path="test_mandelbrot.py", cov_source="mandelbrot", extra_args=None):
    """Run pytest with coverage and parse the output."""
    cmd = [
        sys.executable, "-m", "pytest", test_path,
        f"--cov={cov_source}",
        "--cov-report=term-missing",
        "-v",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout + result.stderr

    passed = _parse_count(stdout, "passed")
    failed = _parse_count(stdout, "failed")
    errors = _parse_count(stdout, "error")
    coverage_pct = _parse_coverage(stdout)

    return {
        "returncode": result.returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "coverage_pct": coverage_pct,
        "stdout": stdout,
    }


def _parse_count(text, keyword):
    """Find the number before keyword in pytest summary, e.g. '3 passed'."""
    match = re.search(rf"(\d+)\s+{re.escape(keyword)}", text)
    return int(match.group(1)) if match else 0


def _parse_coverage(text):
    """Pull total coverage % from the pytest-cov report line."""
    match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", text)
    return float(match.group(1)) if match else None


def build_summary_dataframe(pytest_result):
    """Turn the pytest result dict into a one-row summary dataframe."""
    return pd.DataFrame(
        [
            {
                "passed": pytest_result["passed"],
                "failed": pytest_result["failed"],
                "errors": pytest_result["errors"],
                "coverage_pct": pytest_result["coverage_pct"],
                "all_passed": pytest_result["returncode"] == 0,
            }
        ]
    )


def run_lecture9_study(test_path="test_mandelbrot.py", cov_source="mandelbrot"):
    """Run the test suite and return results + summary dataframe."""
    resolved = str(Path(test_path).resolve())
    pytest_result = run_pytest(test_path=test_path, cov_source=cov_source)
    summary_df = build_summary_dataframe(pytest_result)

    return {
        "pytest_result": pytest_result,
        "summary_df": summary_df,
        "test_path": resolved,
        "cov_source": cov_source,
    }
