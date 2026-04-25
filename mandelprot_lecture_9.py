"""Lecture 9 – Testing & Documentation utilities for the Mandelbrot study.

M1: ``test_mandelbrot.py`` contains the pytest suite.  The helpers here
    run that suite programmatically and surface pass counts and coverage
    percentages for the performance notebook.

M2: Every public function in this module carries full NumPy-style
    docstrings and Python 3.12 type hints, satisfying the documentation
    milestone.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def run_pytest(
    test_path: str = "test_mandelbrot.py",
    cov_source: str = "mandelbrot",
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Execute pytest with optional coverage collection and parse the output.

    Parameters
    ----------
    test_path : str, optional
        Path to the test file or directory (default ``"test_mandelbrot.py"``).
    cov_source : str, optional
        Module or package name passed to ``--cov`` (default ``"mandelbrot"``).
    extra_args : list[str] or None, optional
        Additional arguments forwarded to pytest verbatim.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``returncode`` (*int*) — pytest exit code (0 = all passed).
        - ``passed`` (*int*) — number of passing tests.
        - ``failed`` (*int*) — number of failing tests.
        - ``errors`` (*int*) — number of tests that raised collection errors.
        - ``coverage_pct`` (*float* or *None*) — total coverage percentage
          extracted from the coverage report, or *None* if not available.
        - ``stdout`` (*str*) — raw combined stdout from pytest.
    """
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


def _parse_count(text: str, keyword: str) -> int:
    """Extract a numeric count preceding *keyword* from pytest summary output.

    Parameters
    ----------
    text : str
        Raw stdout text from a pytest run.
    keyword : str
        Keyword to search for, e.g. ``"passed"`` or ``"failed"``.

    Returns
    -------
    int
        The integer count found immediately before *keyword*, or 0 if the
        keyword is not present.
    """
    import re
    match = re.search(rf"(\d+)\s+{re.escape(keyword)}", text)
    return int(match.group(1)) if match else 0


def _parse_coverage(text: str) -> float | None:
    """Extract the total coverage percentage from a pytest-cov report.

    Parameters
    ----------
    text : str
        Raw stdout text from a pytest run that includes a coverage report.

    Returns
    -------
    float or None
        Coverage percentage (0–100) if found, otherwise *None*.
    """
    import re
    match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", text)
    return float(match.group(1)) if match else None


def build_summary_dataframe(pytest_result: dict[str, Any]) -> pd.DataFrame:
    """Convert a :func:`run_pytest` result dict to a one-row summary DataFrame.

    Parameters
    ----------
    pytest_result : dict[str, Any]
        Dictionary returned by :func:`run_pytest`.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns ``passed``, ``failed``,
        ``errors``, ``coverage_pct``, and ``all_passed``.
    """
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


def run_lecture9_study(
    test_path: str = "test_mandelbrot.py",
    cov_source: str = "mandelbrot",
) -> dict[str, Any]:
    """Run the full Lecture 9 study: execute the test suite and collect results.

    Parameters
    ----------
    test_path : str, optional
        Path to the pytest test file (default ``"test_mandelbrot.py"``).
    cov_source : str, optional
        Module name for coverage measurement (default ``"mandelbrot"``).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``pytest_result`` (*dict*) — raw result from :func:`run_pytest`.
        - ``summary_df`` (*pd.DataFrame*) — one-row summary table.
        - ``test_path`` (*str*) — resolved absolute path to the test file.
        - ``cov_source`` (*str*) — coverage source module name.
    """
    resolved = str(Path(test_path).resolve())
    pytest_result = run_pytest(test_path=test_path, cov_source=cov_source)
    summary_df = build_summary_dataframe(pytest_result)

    return {
        "pytest_result": pytest_result,
        "summary_df": summary_df,
        "test_path": resolved,
        "cov_source": cov_source,
    }
