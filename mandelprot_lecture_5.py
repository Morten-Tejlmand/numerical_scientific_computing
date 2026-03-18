import multiprocessing as mp
import os
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

from mandelbrot import (
    compute_mandelbrot_naive_numba,
    compute_mandelbrot_numpy,
    mandelbrot_naive,
)


def bench(fn, *args, runs=3, warmup=True):
    if warmup:
        fn(*args)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)

    return statistics.median(times), min(times), max(times)


@njit
def compute_mandelbrot_chunk_numba(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    row_start,
    row_end,
):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    chunk_height = row_end - row_start
    chunk = np.zeros((chunk_height, width), dtype=np.int32)

    for local_i in range(chunk_height):
        i = row_start + local_i
        c_imag = y[i]

        for j in range(width):
            c_real = x[j]

            z_real = 0.0
            z_imag = 0.0
            n = 0

            while n < max_iter and (z_real * z_real + z_imag * z_imag) <= 4.0:
                new_real = z_real * z_real - z_imag * z_imag + c_real
                new_imag = 2.0 * z_real * z_imag + c_imag
                z_real = new_real
                z_imag = new_imag
                n += 1

            chunk[local_i, j] = n

    return chunk


def split_rows_into_chunks(height, n_chunks):
    n_chunks = max(1, min(int(n_chunks), int(height)))
    edges = np.linspace(0, height, n_chunks + 1, dtype=np.int32)

    chunks = []
    for i in range(n_chunks):
        row_start = int(edges[i])
        row_end = int(edges[i + 1])
        if row_start < row_end:
            chunks.append((row_start, row_end))

    return chunks


def mandelbrot_serial_chunked(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_chunks=1,
):
    chunks = split_rows_into_chunks(height, n_chunks)
    chunk_results = []

    for row_start, row_end in chunks:
        chunk_results.append(
            compute_mandelbrot_chunk_numba(
                xmin,
                xmax,
                ymin,
                ymax,
                width,
                height,
                max_iter,
                row_start,
                row_end,
            )
        )

    return np.vstack(chunk_results)


def mandelbrot_chunk_worker(args):
    (
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        row_start,
        row_end,
    ) = args

    return compute_mandelbrot_chunk_numba(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        row_start,
        row_end,
    )


def mandelbrot_parallel(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_workers=None,
    n_chunks=None,
    pool=None,
):
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    if n_chunks is None:
        n_chunks = n_workers

    chunks = split_rows_into_chunks(height, n_chunks)
    tasks = []
    for row_start, row_end in chunks:
        tasks.append(
            (
                xmin,
                xmax,
                ymin,
                ymax,
                width,
                height,
                max_iter,
                row_start,
                row_end,
            )
        )

    if pool is None:
        with mp.Pool(processes=n_workers) as local_pool:
            chunk_results = local_pool.map(mandelbrot_chunk_worker, tasks)
    else:
        chunk_results = pool.map(mandelbrot_chunk_worker, tasks)

    return np.vstack(chunk_results)


def bench_parallel_config(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_workers,
    n_chunks,
    runs=3,
):
    with mp.Pool(processes=n_workers) as pool:
        mandelbrot_parallel(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_workers=n_workers,
            n_chunks=n_chunks,
            pool=pool,
        )

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            mandelbrot_parallel(
                xmin,
                xmax,
                ymin,
                ymax,
                width,
                height,
                max_iter,
                n_workers=n_workers,
                n_chunks=n_chunks,
                pool=pool,
            )
            times.append(time.perf_counter() - t0)

    return statistics.median(times), min(times), max(times)


def verify_output_matches_serial(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_workers,
    n_chunks,
):
    serial_img = mandelbrot_serial_chunked(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_chunks=n_chunks,
    )
    parallel_img = mandelbrot_parallel(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_workers=n_workers,
        n_chunks=n_chunks,
    )
    return bool(np.array_equal(serial_img, parallel_img))


def benchmark_chunk_configurations(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_workers,
    multipliers=(1, 2, 4, 8, 16),
    runs=3,
):
    rows = []
    for factor in multipliers:
        n_chunks = max(1, int(factor) * int(n_workers))
        t1_median, _, _ = bench_parallel_config(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_workers=1,
            n_chunks=n_chunks,
            runs=runs,
        )
        tp_median, tp_min, tp_max = bench_parallel_config(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_workers=n_workers,
            n_chunks=n_chunks,
            runs=runs,
        )

        speedup = t1_median / tp_median
        lif = max(0.0, n_workers * tp_median / t1_median - 1.0)
        rows.append(
            {
                "n_chunks": n_chunks,
                "time_s": tp_median,
                "min_s": tp_min,
                "max_s": tp_max,
                "t1_s": t1_median,
                "speedup": speedup,
                "LIF": lif,
            }
        )

    best_row = min(rows, key=lambda x: x["LIF"])
    return rows, best_row


def benchmark_speedup_vs_cores(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    best_chunks,
    process_counts,
    runs=3,
):
    t1_median, _, _ = bench(
        mandelbrot_serial_chunked,
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        max(1, best_chunks),
        runs=runs,
    )

    rows = []
    for p in process_counts:
        chunks = max(best_chunks, p)
        tp_median, tp_min, tp_max = bench_parallel_config(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_workers=p,
            n_chunks=chunks,
            runs=runs,
        )

        rows.append(
            {
                "processes": p,
                "n_chunks": chunks,
                "time_s": tp_median,
                "min_s": tp_min,
                "max_s": tp_max,
                "speedup": t1_median / tp_median,
                "ideal_speedup": float(p),
            }
        )

    return t1_median, rows


def estimate_serial_fraction(speedup, p):
    if p <= 1:
        return np.nan
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


def benchmark_method_comparison(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    parallel_workers,
    parallel_chunks,
    runs=3,
):
    naive_t, _, _ = bench(
        mandelbrot_naive,
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        runs=runs,
    )

    numpy_t, _, _ = bench(
        compute_mandelbrot_numpy,
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        runs=runs,
    )

    numba_t, _, _ = bench(
        compute_mandelbrot_naive_numba,
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        runs=runs,
    )

    parallel_t, _, _ = bench_parallel_config(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_workers=parallel_workers,
        n_chunks=parallel_chunks,
        runs=runs,
    )

    rows = [
        {"method": "Naive Python", "time_s": naive_t},
        {"method": "NumPy", "time_s": numpy_t},
        {"method": "Numba", "time_s": numba_t},
        {"method": "Parallel (optimized)", "time_s": parallel_t},
    ]

    for row in rows:
        row["speedup_vs_naive"] = naive_t / row["time_s"]

    return rows


def plot_chunk_time(chunk_rows, output_path="lecture5_chunk_time.png"):
    x = [row["n_chunks"] for row in chunk_rows]
    y = [row["time_s"] for row in chunk_rows]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("Number of chunks")
    plt.ylabel("Median execution time (s)")
    plt.title("Chunk size vs execution time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_speedup(speed_rows, output_path="lecture5_speedup.png"):
    x = [row["processes"] for row in speed_rows]
    actual = [row["speedup"] for row in speed_rows]
    ideal = [row["ideal_speedup"] for row in speed_rows]

    plt.figure(figsize=(7, 4))
    plt.plot(x, actual, marker="o", label="actual speedup")
    plt.plot(x, ideal, marker="o", linestyle="--", label="ideal speedup")
    plt.xlabel("Processes")
    plt.ylabel("Speedup S(p)")
    plt.title("Speedup vs cores")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def run_lecture5_study(
    xmin=-2.0,
    xmax=1.0,
    ymin=-1.5,
    ymax=1.5,
    width=1200,
    height=900,
    max_iter=400,
    n_workers_l04=4,
    process_counts=(1, 2, 4, 8),
    runs=3,
):
    process_counts = [p for p in process_counts if p <= (os.cpu_count() or 1)]
    if not process_counts:
        process_counts = [1]

    compute_mandelbrot_chunk_numba(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        0,
        min(height, 8),
    )

    m1_chunks_default = n_workers_l04
    m1_chunks_4x = 4 * n_workers_l04
    m1_same_image = verify_output_matches_serial(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_workers=n_workers_l04,
        n_chunks=m1_chunks_4x,
    )
    m1_default_t, _, _ = bench_parallel_config(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_workers=n_workers_l04,
        n_chunks=m1_chunks_default,
        runs=runs,
    )
    m1_4x_t, _, _ = bench_parallel_config(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_workers=n_workers_l04,
        n_chunks=m1_chunks_4x,
        runs=runs,
    )

    chunk_rows, best_chunk_row = benchmark_chunk_configurations(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_workers=n_workers_l04,
        multipliers=(1, 2, 4, 8, 16),
        runs=runs,
    )

    best_chunks = int(best_chunk_row["n_chunks"])
    t1_speedup, speed_rows = benchmark_speedup_vs_cores(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        best_chunks=best_chunks,
        process_counts=process_counts,
        runs=runs,
    )

    best_speed_row = max(speed_rows, key=lambda x: x["speedup"])
    serial_fraction = estimate_serial_fraction(
        best_speed_row["speedup"],
        best_speed_row["processes"],
    )

    comparison_rows = benchmark_method_comparison(
        xmin,
        xmax,
        ymin,
        ymax,
        512,
        512,
        200,
        parallel_workers=n_workers_l04,
        parallel_chunks=best_chunks,
        runs=3,
    )

    tracker_time, tracker_min, tracker_max = bench_parallel_config(
        xmin,
        xmax,
        ymin,
        ymax,
        1024,
        1024,
        max_iter,
        n_workers=n_workers_l04,
        n_chunks=best_chunks,
        runs=3,
    )

    return {
        "m1_same_image": m1_same_image,
        "m1_chunks_default": m1_chunks_default,
        "m1_chunks_4x": m1_chunks_4x,
        "m1_time_default": m1_default_t,
        "m1_time_4x": m1_4x_t,
        "chunk_rows": chunk_rows,
        "best_chunk_row": best_chunk_row,
        "speedup_t1": t1_speedup,
        "speed_rows": speed_rows,
        "serial_fraction": float(serial_fraction),
        "comparison_rows": comparison_rows,
        "tracker_1024": {
            "median_s": tracker_time,
            "min_s": tracker_min,
            "max_s": tracker_max,
        },
    }


def to_dataframes(results):
    return (
        pd.DataFrame(results["chunk_rows"]),
        pd.DataFrame(results["speed_rows"]),
        pd.DataFrame(results["comparison_rows"]),
    )
