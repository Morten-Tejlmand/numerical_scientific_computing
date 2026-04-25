"""Lecture 10 – GPU Computing with PyOpenCL: Mandelbrot Set Kernels.

M1: float32 PyOpenCL kernel – compute and save image, record runtime.
M2: float64 PyOpenCL kernel – compare precision against float32 at high iterations.
M3: bar chart comparing GPU kernels against all prior CPU implementations.
"""

from __future__ import annotations

import math
import statistics
import time
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyopencl as cl


_KERNEL_F32 = """
__kernel void mandelbrot_f32(
    __global int *output,
    const float xmin, const float xmax,
    const float ymin, const float ymax,
    const int width, const int height,
    const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= width || row >= height) return;
    float cx = xmin + (float)col * (xmax - xmin) / (float)(width - 1);
    float cy = ymin + (float)row * (ymax - ymin) / (float)(height - 1);
    float zx = 0.0f, zy = 0.0f, zx2, zy2;
    int n = 0;
    while (n < max_iter) {
        zx2 = zx * zx;
        zy2 = zy * zy;
        if (zx2 + zy2 > 4.0f) break;
        zy = 2.0f * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        n++;
    }
    output[row * width + col] = n;
}
"""

_KERNEL_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_f64(
    __global int *output,
    const double xmin, const double xmax,
    const double ymin, const double ymax,
    const int width, const int height,
    const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= width || row >= height) return;
    double cx = xmin + (double)col * (xmax - xmin) / (double)(width - 1);
    double cy = ymin + (double)row * (ymax - ymin) / (double)(height - 1);
    double zx = 0.0, zy = 0.0, zx2, zy2;
    int n = 0;
    while (n < max_iter) {
        zx2 = zx * zx;
        zy2 = zy * zy;
        if (zx2 + zy2 > 4.0) break;
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        n++;
    }
    output[row * width + col] = n;
}
"""


def _select_device() -> cl.Device:
    """Pick the best GPU device, preferring NVIDIA over others."""
    best: cl.Device | None = None
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                if best is None or "NVIDIA" in device.name.upper():
                    best = device
    if best is None:
        # Fallback to any available device
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                return device
        raise RuntimeError("No OpenCL devices found")
    return best


def build_context() -> tuple[cl.Context, cl.CommandQueue, cl.Device]:
    """Create an OpenCL context and command queue on the best available GPU.

    Returns
    -------
    tuple[cl.Context, cl.CommandQueue, cl.Device]
        The context, queue, and selected device.
    """
    device = _select_device()
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    return ctx, queue, device


def supports_float64(device: cl.Device) -> bool:
    """Return True if *device* supports the cl_khr_fp64 extension.

    Parameters
    ----------
    device : cl.Device
        The OpenCL device to inspect.

    Returns
    -------
    bool
        Whether double-precision floating-point is available.
    """
    return "cl_khr_fp64" in device.extensions.split()


def _run_kernel(
    queue: cl.CommandQueue,
    prg: cl.Program,
    kernel_name: str,
    scalar_dtype: type,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    width: int, height: int,
    max_iter: int,
) -> np.ndarray:
    """Execute a compiled Mandelbrot kernel and return the iteration grid."""
    ctx = queue.context
    output = np.empty(height * width, dtype=np.int32)
    buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
    kernel = cl.Kernel(prg, kernel_name)
    kernel(
        queue, (width, height), None,
        buf,
        scalar_dtype(xmin), scalar_dtype(xmax),
        scalar_dtype(ymin), scalar_dtype(ymax),
        np.int32(width), np.int32(height),
        np.int32(max_iter),
    )
    cl.enqueue_copy(queue, output, buf)
    queue.finish()
    return output.reshape(height, width)


def mandelbrot_opencl_f32(
    ctx: cl.Context,
    queue: cl.CommandQueue,
    prg_f32: cl.Program,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    width: int, height: int,
    max_iter: int,
) -> np.ndarray:
    """Compute the Mandelbrot set with a float32 OpenCL kernel.

    Parameters
    ----------
    ctx : cl.Context
        The OpenCL context.
    queue : cl.CommandQueue
        The command queue.
    prg_f32 : cl.Program
        Pre-compiled float32 program.
    xmin, xmax, ymin, ymax : float
        Bounds of the complex plane.
    width, height : int
        Output grid dimensions.
    max_iter : int
        Maximum iteration count.

    Returns
    -------
    np.ndarray
        2-D int32 array of shape (height, width) with iteration counts.
    """
    return _run_kernel(queue, prg_f32, "mandelbrot_f32", np.float32,
                       xmin, xmax, ymin, ymax, width, height, max_iter)


def mandelbrot_opencl_f64(
    ctx: cl.Context,
    queue: cl.CommandQueue,
    prg_f64: cl.Program,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    width: int, height: int,
    max_iter: int,
) -> np.ndarray:
    """Compute the Mandelbrot set with a float64 OpenCL kernel.

    Parameters
    ----------
    ctx : cl.Context
        The OpenCL context.
    queue : cl.CommandQueue
        The command queue.
    prg_f64 : cl.Program
        Pre-compiled float64 program.
    xmin, xmax, ymin, ymax : float
        Bounds of the complex plane.
    width, height : int
        Output grid dimensions.
    max_iter : int
        Maximum iteration count.

    Returns
    -------
    np.ndarray
        2-D int32 array of shape (height, width) with iteration counts.
    """
    return _run_kernel(queue, prg_f64, "mandelbrot_f64", np.float64,
                       xmin, xmax, ymin, ymax, width, height, max_iter)


def bench(fn, *args, runs: int = 3, warmup: bool = True) -> tuple[float, float, float]:
    """Measure wall-clock time of *fn(*args)* including any host↔GPU transfers.

    Parameters
    ----------
    fn : callable
        Function to benchmark.
    *args :
        Positional arguments forwarded to *fn*.
    runs : int, optional
        Number of timed repetitions (default 3).
    warmup : bool, optional
        If True, run *fn* once before timing to warm up JIT / GPU caches.

    Returns
    -------
    tuple[float, float, float]
        (median, min, max) wall-clock times in seconds.
    """
    if warmup:
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def plot_mandelbrot_image(
    image: np.ndarray,
    title: str,
    output_path: str = "lecture10_f32.png",
) -> str:
    """Save a false-colour Mandelbrot iteration-count image.

    Parameters
    ----------
    image : np.ndarray
        2-D int32 iteration grid.
    title : str
        Plot title.
    output_path : str, optional
        File path for the saved PNG.

    Returns
    -------
    str
        The path to the saved image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="inferno", origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precision_comparison(
    img_f32: np.ndarray,
    img_f64: np.ndarray,
    output_path: str = "lecture10_precision_comparison.png",
) -> str:
    """Save a three-panel figure: float32 image, float64 image, pixel difference.

    Parameters
    ----------
    img_f32 : np.ndarray
        Iteration grid from the float32 kernel.
    img_f64 : np.ndarray
        Iteration grid from the float64 kernel.
    output_path : str, optional
        File path for the saved PNG.

    Returns
    -------
    str
        The path to the saved image.
    """
    diff = np.abs(img_f64.astype(np.int32) - img_f32.astype(np.int32))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(img_f32, cmap="inferno", origin="lower", aspect="auto")
    axes[0].set_title("GPU float32")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(img_f64, cmap="inferno", origin="lower", aspect="auto")
    axes[1].set_title("GPU float64")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, cmap="hot", origin="lower", aspect="auto")
    axes[2].set_title("Pixel difference |f64 \u2212 f32|")
    fig.colorbar(im2, ax=axes[2])

    plt.suptitle(
        "Lecture 10 \u2013 M2: float32 vs float64 at high iterations",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_benchmark_bar(
    method_names: list[str],
    times_s: list[float],
    width: int,
    height: int,
    max_iter: int,
    output_path: str = "lecture10_benchmark.png",
) -> str:
    """Save a log-scale bar chart comparing runtimes across all implementations.

    Parameters
    ----------
    method_names : list[str]
        Labels for each bar.
    times_s : list[float]
        Corresponding median runtimes in seconds.
    width, height, max_iter : int
        Grid parameters shown in the title.
    output_path : str, optional
        File path for the saved PNG.

    Returns
    -------
    str
        The path to the saved image.
    """
    colors = []
    for name in method_names:
        if "GPU" in name and "float32" in name:
            colors.append("#2ca02c")
        elif "GPU" in name and "float64" in name:
            colors.append("#8c564b")
        elif "Naive" in name:
            colors.append("#d62728")
        elif "NumPy" in name:
            colors.append("#1f77b4")
        else:
            colors.append("#ff7f0e")

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(method_names, times_s, color=colors, edgecolor="black", linewidth=0.8)

    for bar, t in zip(bars, times_s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{t * 1000:.2f} ms",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Time (s, log scale)")
    ax.set_yscale("log")
    ax.set_title(
        f"Lecture 10 \u2013 M3: CPU vs GPU runtime ({width}\u00d7{height}, {max_iter} iter)"
    )
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def update_tracker(
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    width: int, height: int,
    max_iter: int,
    gpu_f32_time: float,
    gpu_f64_time: float,
    naive_time: float,
    tracker_path: str = "performance_tracker.csv",
) -> tuple[str, pd.DataFrame]:
    """Append GPU benchmark results for lecture_10 to the performance tracker CSV.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Complex-plane bounds used for benchmarking.
    width, height, max_iter : int
        Grid dimensions and iteration cap.
    gpu_f32_time, gpu_f64_time : float
        Median runtimes in seconds for the two GPU kernels.
    naive_time : float
        Naive Python baseline time used to compute speedup.
    tracker_path : str, optional
        Path to the CSV performance tracker.

    Returns
    -------
    tuple[str, pd.DataFrame]
        The tracker path and a DataFrame of the newly appended rows.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict] = [
        {
            "timestamp": now, "lecture": "lecture_10",
            "width": width, "height": height, "max_iter": max_iter,
            "method": "GPU float32 (OpenCL)",
            "time_s": gpu_f32_time,
            "speedup_vs_naive": naive_time / gpu_f32_time,
        },
    ]
    if not math.isnan(gpu_f64_time):
        rows.append({
            "timestamp": now, "lecture": "lecture_10",
            "width": width, "height": height, "max_iter": max_iter,
            "method": "GPU float64 (OpenCL)",
            "time_s": gpu_f64_time,
            "speedup_vs_naive": naive_time / gpu_f64_time,
        })

    try:
        existing = pd.read_csv(tracker_path)
        updated = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    except FileNotFoundError:
        updated = pd.DataFrame(rows)

    updated.to_csv(tracker_path, index=False)
    return tracker_path, pd.DataFrame(rows)


def run_lecture10_study(
    xmin: float = -2.0, xmax: float = 1.0,
    ymin: float = -1.5, ymax: float = 1.5,
    width: int = 800, height: int = 800,
    max_iter: int = 256,
    tracker_width: int = 256, tracker_height: int = 256,
    tracker_max_iter: int = 120,
    tracker_path: str = "performance_tracker.csv",
) -> dict[str, Any]:
    """Run the full Lecture 10 GPU Mandelbrot study.

    Milestones
    ----------
    M1 : Build a float32 PyOpenCL kernel, compute a full-resolution image, and
         record the runtime.
    M2 : Build a float64 kernel and compare precision visually against float32
         at *max_iter* iterations.
    M3 : Benchmark both GPU kernels at the same grid size as prior CPU milestones
         and produce a single bar chart spanning all implementations.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Complex-plane extent for the visualisation renders.
    width, height : int
        Resolution for the M1/M2 images.
    max_iter : int
        Iteration cap for M1/M2 images.
    tracker_width, tracker_height, tracker_max_iter : int
        Grid parameters matching previous CPU benchmarks.
    tracker_path : str, optional
        Path to the CSV performance tracker.

    Returns
    -------
    dict[str, Any]
        All results: images, plot paths, timings, and updated tracker rows.
    """
    ctx, queue, device = build_context()
    device_name = device.name.strip()
    has_f64 = supports_float64(device)

    # Pre-compile kernels once (excluded from benchmark timing)
    prg_f32 = cl.Program(ctx, _KERNEL_F32).build()
    prg_f64 = cl.Program(ctx, _KERNEL_F64).build() if has_f64 else None

    # --- M1: float32 kernel at full resolution ---
    img_f32 = mandelbrot_opencl_f32(
        ctx, queue, prg_f32, xmin, xmax, ymin, ymax, width, height, max_iter
    )
    m1_plot = plot_mandelbrot_image(
        img_f32,
        f"Lecture 10 \u2013 M1: GPU float32 ({device_name})",
        "lecture10_f32.png",
    )

    # --- M2: precision comparison ---
    if has_f64 and prg_f64 is not None:
        img_f64 = mandelbrot_opencl_f64(
            ctx, queue, prg_f64, xmin, xmax, ymin, ymax, width, height, max_iter
        )
    else:
        # Fall back to CPU float64 if GPU does not support double precision
        from mandelbrot import compute_mandelbrot_numba_float64
        img_f64 = compute_mandelbrot_numba_float64(
            xmin, xmax, ymin, ymax, width, height, max_iter
        )

    m2_plot = plot_precision_comparison(img_f32, img_f64, "lecture10_precision_comparison.png")

    # --- Benchmark at tracker grid size (M3 data) ---
    f32_fn = lambda: mandelbrot_opencl_f32(  # noqa: E731
        ctx, queue, prg_f32, xmin, xmax, ymin, ymax,
        tracker_width, tracker_height, tracker_max_iter,
    )
    f32_t, f32_min, f32_max = bench(f32_fn, runs=3, warmup=True)

    if has_f64 and prg_f64 is not None:
        f64_fn = lambda: mandelbrot_opencl_f64(  # noqa: E731
            ctx, queue, prg_f64, xmin, xmax, ymin, ymax,
            tracker_width, tracker_height, tracker_max_iter,
        )
        f64_t, f64_min, f64_max = bench(f64_fn, runs=3, warmup=True)
    else:
        f64_t = f64_min = f64_max = float("nan")

    # --- Load CPU reference times from tracker (latest lecture_8 run) ---
    tracker_df = pd.read_csv(tracker_path)
    lec8 = tracker_df[
        (tracker_df["lecture"] == "lecture_8")
        & (tracker_df["width"] == tracker_width)
    ].copy()
    cpu_ref = (
        lec8.sort_values("timestamp")
        .drop_duplicates("method", keep="last")
    )
    cpu_ref_dict: dict[str, float] = dict(zip(cpu_ref["method"], cpu_ref["time_s"]))
    naive_t = cpu_ref_dict.get("Naive Python", 0.589)

    # Update tracker with GPU results
    _, new_rows = update_tracker(
        xmin, xmax, ymin, ymax,
        tracker_width, tracker_height, tracker_max_iter,
        f32_t,
        f64_t if not math.isnan(f64_t) else float("nan"),
        naive_t,
        tracker_path=tracker_path,
    )

    # --- Build M3 bar chart data ---
    method_names: list[str] = []
    times_s: list[float] = []
    for method in ["Naive Python", "NumPy", "Numba float64", "Numba float32"]:
        if method in cpu_ref_dict:
            method_names.append(method)
            times_s.append(cpu_ref_dict[method])
    method_names.append("GPU float32 (OpenCL)")
    times_s.append(f32_t)
    if has_f64 and not math.isnan(f64_t):
        method_names.append("GPU float64 (OpenCL)")
        times_s.append(f64_t)

    m3_plot = plot_benchmark_bar(
        method_names, times_s,
        tracker_width, tracker_height, tracker_max_iter,
        "lecture10_benchmark.png",
    )

    return {
        "device_name": device_name,
        "has_f64": has_f64,
        "img_f32": img_f32,
        "img_f64": img_f64,
        "m1_plot": m1_plot,
        "m2_plot": m2_plot,
        "m3_plot": m3_plot,
        "gpu_f32_time_s": f32_t,
        "gpu_f32_min_s": f32_min,
        "gpu_f32_max_s": f32_max,
        "gpu_f64_time_s": f64_t,
        "gpu_f64_min_s": f64_min,
        "gpu_f64_max_s": f64_max,
        "cpu_ref_dict": cpu_ref_dict,
        "method_names": method_names,
        "times_s": times_s,
        "new_tracker_rows": new_rows,
    }


def to_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Convert run_lecture10_study results to a flat summary DataFrame.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary returned by :func:`run_lecture10_study`.

    Returns
    -------
    pd.DataFrame
        One row per implementation with columns ``method`` and ``time_s``.
    """
    rows = [
        {"method": name, "time_s": round(t, 6)}
        for name, t in zip(results["method_names"], results["times_s"])
    ]
    return pd.DataFrame(rows)
