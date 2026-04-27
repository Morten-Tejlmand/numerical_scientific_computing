"""Lecture 10 – GPU Computing with PyOpenCL: Mandelbrot Set Kernels.

M1: float32 PyOpenCL kernel – compute and save image, record runtime.
M2: float64 PyOpenCL kernel – compare precision against float32 at high iterations.
M3: bar chart comparing GPU kernels against all prior CPU implementations.
"""

import math
import statistics
import time
from datetime import datetime

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


def _select_device():
    """Pick the best GPU, preferring NVIDIA."""
    best = None
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                if best is None or "NVIDIA" in device.name.upper():
                    best = device
    if best is None:
        # just grab whatever is available
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                return device
        raise RuntimeError("No OpenCL devices found")
    return best


def build_context():
    """Create an OpenCL context and command queue on the best available GPU."""
    device = _select_device()
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    return ctx, queue, device


def supports_float64(device):
    """Check if the device has the cl_khr_fp64 extension."""
    return "cl_khr_fp64" in device.extensions.split()


def _run_kernel(queue, prg, kernel_name, scalar_dtype, xmin, xmax, ymin, ymax, width, height, max_iter):
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


def mandelbrot_opencl_f32(ctx, queue, prg_f32, xmin, xmax, ymin, ymax, width, height, max_iter):
    """Compute Mandelbrot with the float32 OpenCL kernel."""
    return _run_kernel(queue, prg_f32, "mandelbrot_f32", np.float32,
                       xmin, xmax, ymin, ymax, width, height, max_iter)


def mandelbrot_opencl_f64(ctx, queue, prg_f64, xmin, xmax, ymin, ymax, width, height, max_iter):
    """Compute Mandelbrot with the float64 OpenCL kernel."""
    return _run_kernel(queue, prg_f64, "mandelbrot_f64", np.float64,
                       xmin, xmax, ymin, ymax, width, height, max_iter)


def bench(fn, *args, runs=3, warmup=True):
    """Time fn(*args) and return (median, min, max) in seconds."""
    if warmup:
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def plot_mandelbrot_image(image, title, output_path="lecture10_f32.png"):
    """Save a false-colour Mandelbrot image."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="inferno", origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precision_comparison(img_f32, img_f64, output_path="lecture10_precision_comparison.png"):
    """Three-panel figure: float32, float64, and pixel difference."""
    diff = np.abs(img_f64.astype(np.int32) - img_f32.astype(np.int32))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(img_f32, cmap="inferno", origin="lower", aspect="auto")
    axes[0].set_title("GPU float32")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(img_f64, cmap="inferno", origin="lower", aspect="auto")
    axes[1].set_title("GPU float64")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, cmap="hot", origin="lower", aspect="auto")
    axes[2].set_title("Pixel difference |f64 − f32|")
    fig.colorbar(im2, ax=axes[2])

    plt.suptitle(
        "Lecture 10 – M2: float32 vs float64 at high iterations",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_benchmark_bar(method_names, times_s, width, height, max_iter, output_path="lecture10_benchmark.png"):
    """Log-scale bar chart comparing runtimes across all implementations."""
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
        f"Lecture 10 – M3: CPU vs GPU runtime ({width}×{height}, {max_iter} iter)"
    )
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def update_tracker(xmin, xmax, ymin, ymax, width, height, max_iter,
                   gpu_f32_time, gpu_f64_time, naive_time,
                   tracker_path="performance_tracker.csv"):
    """Append GPU benchmark rows to the performance tracker CSV."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
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
    xmin=-2.0, xmax=1.0,
    ymin=-1.5, ymax=1.5,
    width=800, height=800,
    max_iter=256,
    tracker_width=256, tracker_height=256,
    tracker_max_iter=120,
    tracker_path="performance_tracker.csv",
):
    """Run the full Lecture 10 GPU Mandelbrot study (M1, M2, M3)."""
    ctx, queue, device = build_context()
    device_name = device.name.strip()
    has_f64 = supports_float64(device)

    # compile kernels
    prg_f32 = cl.Program(ctx, _KERNEL_F32).build()
    prg_f64 = cl.Program(ctx, _KERNEL_F64).build() if has_f64 else None

    # M1: float32 image
    img_f32 = mandelbrot_opencl_f32(
        ctx, queue, prg_f32, xmin, xmax, ymin, ymax, width, height, max_iter
    )
    m1_plot = plot_mandelbrot_image(
        img_f32,
        f"Lecture 10 – M1: GPU float32 ({device_name})",
        "lecture10_f32.png",
    )

    # M2: compare float32 vs float64
    if has_f64 and prg_f64 is not None:
        img_f64 = mandelbrot_opencl_f64(
            ctx, queue, prg_f64, xmin, xmax, ymin, ymax, width, height, max_iter
        )
    else:
        # GPU doesnt support float64, fall back to CPU
        from mandelbrot import compute_mandelbrot_numba_float64
        img_f64 = compute_mandelbrot_numba_float64(
            xmin, xmax, ymin, ymax, width, height, max_iter
        )

    m2_plot = plot_precision_comparison(img_f32, img_f64, "lecture10_precision_comparison.png")

    # M3: benchmark at the tracker grid size
    def f32_fn():
        return mandelbrot_opencl_f32(
            ctx, queue, prg_f32, xmin, xmax, ymin, ymax,
            tracker_width, tracker_height, tracker_max_iter,
        )
    f32_t, f32_min, f32_max = bench(f32_fn, runs=3, warmup=True)

    if has_f64 and prg_f64 is not None:
        def f64_fn():
            return mandelbrot_opencl_f64(
                ctx, queue, prg_f64, xmin, xmax, ymin, ymax,
                tracker_width, tracker_height, tracker_max_iter,
            )
        f64_t, f64_min, f64_max = bench(f64_fn, runs=3, warmup=True)
    else:
        f64_t = f64_min = f64_max = float("nan")

    # load CPU reference times from tracker
    tracker_df = pd.read_csv(tracker_path)
    lec8 = tracker_df[
        (tracker_df["lecture"] == "lecture_8")
        & (tracker_df["width"] == tracker_width)
    ].copy()
    cpu_ref = lec8.sort_values("timestamp").drop_duplicates("method", keep="last")
    cpu_ref_dict = dict(zip(cpu_ref["method"], cpu_ref["time_s"]))
    naive_t = cpu_ref_dict.get("Naive Python", 0.589)

    _, new_rows = update_tracker(
        xmin, xmax, ymin, ymax,
        tracker_width, tracker_height, tracker_max_iter,
        f32_t,
        f64_t if not math.isnan(f64_t) else float("nan"),
        naive_t,
        tracker_path=tracker_path,
    )

    # build bar chart data
    method_names = []
    times_s = []
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


def to_dataframe(results):
    """Convert run_lecture10_study results to a flat summary dataframe."""
    rows = [
        {"method": name, "time_s": round(t, 6)}
        for name, t in zip(results["method_names"], results["times_s"])
    ]
    return pd.DataFrame(rows)
