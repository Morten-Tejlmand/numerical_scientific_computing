import statistics
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bench(fn, *args, runs=3, warmup=True):
    if warmup:
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def trajectory_divergence(c, max_iter=256):
    """Track |z64 - z32| error per iteration for a single point c."""
    c64 = np.complex128(c)
    c32 = np.complex64(c)
    z64 = np.complex128(0.0)
    z32 = np.complex64(0.0)

    path64 = []
    path32 = []
    errors = []

    for _ in range(max_iter):
        z64 = z64 * z64 + c64
        z32 = z32 * z32 + c32
        path64.append(z64)
        path32.append(z32)
        errors.append(abs(complex(z64) - complex(z32)))
        if abs(z64) > 2.0 or abs(z32) > 2.0:
            break

    return np.array(path64), np.array(path32), np.array(errors)


def find_first_divergence_iter(errors, epsilon=1e-6):
    """First iteration index where error exceeds epsilon. Returns len(errors) if never."""
    above = np.where(errors > epsilon)[0]
    return int(above[0]) if len(above) > 0 else len(errors)


def compute_sensitivity_map(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Accumulate |z64 - z32| per pixel over all iterations. High values = boundary."""
    x64 = np.linspace(xmin, xmax, width, dtype=np.float64)
    y64 = np.linspace(ymin, ymax, height, dtype=np.float64)
    X64, Y64 = np.meshgrid(x64, y64)
    C64 = X64 + 1j * Y64

    x32 = np.linspace(xmin, xmax, width, dtype=np.float32)
    y32 = np.linspace(ymin, ymax, height, dtype=np.float32)
    X32, Y32 = np.meshgrid(x32, y32)
    C32 = (X32 + 1j * Y32).astype(np.complex64)

    z64 = np.zeros_like(C64)
    z32 = np.zeros(C64.shape, dtype=np.complex64)
    accumulated = np.zeros(C64.shape, dtype=np.float64)
    alive = np.ones(C64.shape, dtype=bool)

    for _ in range(max_iter):
        z64 = z64 * z64 + C64
        z32 = z32 * z32 + C32
        alive &= (np.abs(z64) <= 2.0) & (np.abs(z32) <= 2.0)
        diff = np.nan_to_num(np.abs(z64 - z32.astype(np.complex128)), nan=0.0, posinf=0.0)
        accumulated += diff * alive

    return accumulated


def plot_trajectory_divergence(
    errors, first_drift_iter=None, output_path="lecture8_trajectory.png"
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(errors, color="tab:blue", label="|z₆₄ − z₃₂|")
    if first_drift_iter is not None and first_drift_iter < len(errors):
        ax.axvline(
            first_drift_iter,
            color="tab:red",
            linestyle="--",
            label=f"first drift (iter {first_drift_iter})",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error  (log scale)")
    ax.set_title("Lecture 8 – M1: trajectory divergence (float64 vs float32)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sensitivity_map(
    sensitivity, xmin, xmax, ymin, ymax, output_path="lecture8_sensitivity.png"
):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        np.log1p(sensitivity),
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap="hot",
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="log(1 + accumulated |z₆₄ − z₃₂|)")
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    ax.set_title("Lecture 8 – M2: precision sensitivity map")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def update_tracker(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    runs=3,
    tracker_path="performance_tracker.csv",
):
    from mandelbrot import (
        compute_mandelbrot_numba_float32,
        compute_mandelbrot_numba_float64,
        mandelbrot_naive,
    )

    compute_mandelbrot_numba_float64(xmin, xmax, ymin, ymax, width, height, max_iter)
    compute_mandelbrot_numba_float32(xmin, xmax, ymin, ymax, width, height, max_iter)

    naive_t, _, _ = bench(
        mandelbrot_naive, xmin, xmax, ymin, ymax, width, height, max_iter,
        runs=runs, warmup=False,
    )
    f64_t, _, _ = bench(
        compute_mandelbrot_numba_float64, xmin, xmax, ymin, ymax, width, height, max_iter,
        runs=runs, warmup=False,
    )
    f32_t, _, _ = bench(
        compute_mandelbrot_numba_float32, xmin, xmax, ymin, ymax, width, height, max_iter,
        runs=runs, warmup=False,
    )
    sens_t, _, _ = bench(
        compute_sensitivity_map, xmin, xmax, ymin, ymax, width, height, max_iter,
        runs=runs, warmup=True,
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        {
            "timestamp": now, "lecture": "lecture_8",
            "width": width, "height": height, "max_iter": max_iter,
            "method": "Naive Python", "time_s": naive_t, "speedup_vs_naive": 1.0,
        },
        {
            "timestamp": now, "lecture": "lecture_8",
            "width": width, "height": height, "max_iter": max_iter,
            "method": "Numba float64", "time_s": f64_t, "speedup_vs_naive": naive_t / f64_t,
        },
        {
            "timestamp": now, "lecture": "lecture_8",
            "width": width, "height": height, "max_iter": max_iter,
            "method": "Numba float32", "time_s": f32_t, "speedup_vs_naive": naive_t / f32_t,
        },
        {
            "timestamp": now, "lecture": "lecture_8",
            "width": width, "height": height, "max_iter": max_iter,
            "method": "Sensitivity Map (float32+float64)", "time_s": sens_t,
            "speedup_vs_naive": naive_t / sens_t,
        },
    ]

    try:
        existing = pd.read_csv(tracker_path)
        updated = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    except FileNotFoundError:
        updated = pd.DataFrame(rows)

    updated.to_csv(tracker_path, index=False)
    return tracker_path, pd.DataFrame(rows)


def run_lecture8_study(
    xmin=-2.0,
    xmax=1.0,
    ymin=-1.5,
    ymax=1.5,
    width=800,
    height=800,
    max_iter=256,
    probe_point=-0.7269 + 0.1889j,
    drift_epsilon=1e-6,
    tracker_width=256,
    tracker_height=256,
    tracker_max_iter=120,
    tracker_path="performance_tracker.csv",
):
    # part 1: trajectory divergence for a single point
    path64, path32, errors = trajectory_divergence(probe_point, max_iter)
    first_drift = find_first_divergence_iter(errors, drift_epsilon)

    # part 2: sensitivity map over the whole grid
    t0 = time.perf_counter()
    sensitivity = compute_sensitivity_map(xmin, xmax, ymin, ymax, width, height, max_iter)
    sens_time_s = time.perf_counter() - t0

    traj_plot = plot_trajectory_divergence(errors, first_drift, "lecture8_trajectory.png")
    sens_plot = plot_sensitivity_map(
        sensitivity, xmin, xmax, ymin, ymax, "lecture8_sensitivity.png"
    )

    tracker_path_out, tracker_df = update_tracker(
        xmin, xmax, ymin, ymax,
        tracker_width, tracker_height, tracker_max_iter,
        runs=3,
        tracker_path=tracker_path,
    )

    return {
        "probe_point": str(probe_point),
        "max_iter": max_iter,
        "trajectory_length": len(errors),
        "first_drift_iter": first_drift,
        "max_error": float(errors.max()) if len(errors) > 0 else 0.0,
        "sensitivity_map": sensitivity,
        "sensitivity_max": float(sensitivity.max()),
        "sensitivity_mean": float(sensitivity.mean()),
        "sensitivity_time_s": sens_time_s,
        "trajectory_plot": traj_plot,
        "sensitivity_plot": sens_plot,
        "tracker_df": tracker_df,
        "grid": {"width": width, "height": height},
    }


def to_dataframe(results):
    return pd.DataFrame(
        [
            {
                "probe_point": results["probe_point"],
                "trajectory_length": results["trajectory_length"],
                "first_drift_iter": results["first_drift_iter"],
                "max_error": results["max_error"],
                "sensitivity_max": results["sensitivity_max"],
                "sensitivity_mean": results["sensitivity_mean"],
                "sensitivity_time_s": results["sensitivity_time_s"],
            }
        ]
    )
