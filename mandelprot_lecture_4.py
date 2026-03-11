import multiprocessing as mp
import os
import statistics
import time

import numpy as np
from numba import njit


def bench(fn, *args, runs=5, warmup=True):
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
    """
    Compute Mandelbrot values for rows [row_start, row_end).
    """
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


def compute_mandelbrot_numba_chunked(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    chunk_rows=64,
):
    """
    Sequential chunked wrapper around the Numba chunk kernel.
    """
    result = np.zeros((height, width), dtype=np.int32)

    for row_start in range(0, height, chunk_rows):
        row_end = min(row_start + chunk_rows, height)
        result[row_start:row_end, :] = compute_mandelbrot_chunk_numba(
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

    return result


def mandelbrot_chunk_worker(args):
    """
    Worker used by multiprocessing.Pool.map.
    Must be top-level for Windows pickling.
    """
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

    chunk = compute_mandelbrot_chunk_numba(
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
    return row_start, row_end, chunk


def compute_mandelbrot_pool_map(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    chunk_rows=64,
    n_processes=None,
):
    """
    Parallel Mandelbrot using Pool.map over row chunks.
    """
    if n_processes is None:
        n_processes = os.cpu_count() or 1

    result = np.zeros((height, width), dtype=np.int32)

    tasks = []
    for row_start in range(0, height, chunk_rows):
        row_end = min(row_start + chunk_rows, height)
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

    with mp.Pool(processes=n_processes) as pool:
        mapped = pool.map(mandelbrot_chunk_worker, tasks)

    for row_start, row_end, chunk in mapped:
        result[row_start:row_end, :] = chunk

    return result


def benchmark_mandelbrot_all(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    chunk_rows=64,
    process_counts=None,
    runs=5,
):
    """
    Benchmark sequential chunked Mandelbrot against Pool.map parallel execution.
    """
    if process_counts is None:
        process_counts = [1, 2, 4, 8]

    # Warm up Numba JIT once
    compute_mandelbrot_chunk_numba(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        0,
        min(chunk_rows, height),
    )

    # Sequential baseline
    baseline_median, baseline_min, baseline_max = bench(
        compute_mandelbrot_numba_chunked,
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        chunk_rows,
        runs=runs,
        warmup=False,
    )

    rows = []

    # Parallel benchmarks
    for p in process_counts:
        median_t, min_t, max_t = bench(
            compute_mandelbrot_pool_map,
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            chunk_rows,
            p,
            runs=runs,
            warmup=False,
        )

        speedup = baseline_median / median_t
        efficiency = speedup / p

        rows.append(
            {
                "processes": p,
                "median_time_s": median_t,
                "min_time_s": min_t,
                "max_time_s": max_t,
                "speedup": speedup,
                "efficiency": efficiency,
            }
        )

    return baseline_median, rows
