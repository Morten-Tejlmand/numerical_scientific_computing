import logging
import os
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask import delayed
from dask.distributed import Client, LocalCluster

from mandelbrot import (
    compute_mandelbrot_naive_numba,
    compute_mandelbrot_numpy,
    mandelbrot_naive,
)
from mandelprot_lecture_5 import (
    bench_parallel_config,
    compute_mandelbrot_chunk_numba,
    split_rows_into_chunks,
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


def build_dask_chunk_graph(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_chunks,
):
    chunks = split_rows_into_chunks(height, n_chunks)
    delayed_chunks = []

    for row_start, row_end in chunks:
        delayed_chunks.append(
            delayed(compute_mandelbrot_chunk_numba)(
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

    return chunks, delayed_chunks


def compute_mandelbrot_dask_local(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_chunks,
    client,
):
    _, delayed_chunks = build_dask_chunk_graph(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_chunks,
    )

    futures = client.compute(delayed_chunks)
    chunk_results = client.gather(futures)
    return np.vstack(chunk_results)


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


def open_local_cluster(n_workers=None, threads_per_worker=1):
    logging.getLogger("distributed").setLevel(logging.ERROR)
    logging.getLogger("dask").setLevel(logging.ERROR)

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=False,
        dashboard_address=":0",
        silence_logs="error",
    )
    client = Client(cluster)
    return cluster, client


def benchmark_dask_config(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_chunks,
    client,
    runs=3,
):
    compute_mandelbrot_dask_local(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_chunks,
        client,
    )

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        compute_mandelbrot_dask_local(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_chunks,
            client,
        )
        times.append(time.perf_counter() - t0)

    return statistics.median(times), min(times), max(times)


def milestone1_dask_local(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_workers=4,
    n_chunks=16,
    runs=3,
):
    cluster, client = open_local_cluster(n_workers=n_workers, threads_per_worker=1)
    try:
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
        dask_img = compute_mandelbrot_dask_local(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_chunks=n_chunks,
            client=client,
        )

        same_image = bool(np.array_equal(serial_img, dask_img))
        serial_t, _, _ = bench(
            mandelbrot_serial_chunked,
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_chunks,
            runs=runs,
        )
        dask_t, dask_min, dask_max = benchmark_dask_config(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_chunks=n_chunks,
            client=client,
            runs=runs,
        )

        _, delayed_chunks = build_dask_chunk_graph(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_chunks,
        )
        graph_task_count = len(delayed_chunks)

        return {
            "same_image": same_image,
            "serial_time_s": serial_t,
            "dask_time_s": dask_t,
            "dask_min_s": dask_min,
            "dask_max_s": dask_max,
            "speedup_vs_serial": serial_t / dask_t,
            "graph_task_count": graph_task_count,
            "dashboard_link": client.dashboard_link,
        }
    finally:
        client.close()
        cluster.close()


def benchmark_dask_chunk_sweep(
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
    chunk_candidates = [max(1, int(m) * int(n_workers)) for m in multipliers]

    cluster_1, client_1 = open_local_cluster(n_workers=1, threads_per_worker=1)
    cluster_p, client_p = open_local_cluster(n_workers=n_workers, threads_per_worker=1)

    rows = []
    try:
        for n_chunks in chunk_candidates:
            t1_median, _, _ = benchmark_dask_config(
                xmin,
                xmax,
                ymin,
                ymax,
                width,
                height,
                max_iter,
                n_chunks=n_chunks,
                client=client_1,
                runs=runs,
            )
            tp_median, tp_min, tp_max = benchmark_dask_config(
                xmin,
                xmax,
                ymin,
                ymax,
                width,
                height,
                max_iter,
                n_chunks=n_chunks,
                client=client_p,
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
    finally:
        client_1.close()
        cluster_1.close()
        client_p.close()
        cluster_p.close()

    best_time_row = min(rows, key=lambda x: x["time_s"])
    best_lif_row = min(rows, key=lambda x: x["LIF"])
    return rows, best_time_row, best_lif_row


def benchmark_full_method_comparison(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    mp_workers,
    mp_chunks,
    dask_workers,
    dask_chunks,
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
    mp_t, _, _ = bench_parallel_config(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_workers=mp_workers,
        n_chunks=mp_chunks,
        runs=runs,
    )

    cluster, client = open_local_cluster(n_workers=dask_workers, threads_per_worker=1)
    try:
        dask_t, _, _ = benchmark_dask_config(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_chunks=dask_chunks,
            client=client,
            runs=runs,
        )
    finally:
        client.close()
        cluster.close()

    rows = [
        {"method": "Naive Python", "time_s": naive_t},
        {"method": "NumPy", "time_s": numpy_t},
        {"method": "Numba", "time_s": numba_t},
        {"method": "Numba + multiprocessing", "time_s": mp_t},
        {"method": "Dask (LocalCluster)", "time_s": dask_t},
    ]

    for row in rows:
        row["speedup_vs_naive"] = naive_t / row["time_s"]

    return rows


def append_performance_tracker(
    comparison_rows,
    width,
    height,
    max_iter,
    tracker_path="performance_tracker.csv",
):
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    append_rows = []
    for row in comparison_rows:
        append_rows.append(
            {
                "timestamp": stamp,
                "lecture": "lecture_6",
                "width": int(width),
                "height": int(height),
                "max_iter": int(max_iter),
                "method": row["method"],
                "time_s": float(row["time_s"]),
                "speedup_vs_naive": float(row["speedup_vs_naive"]),
            }
        )

    new_df = pd.DataFrame(append_rows)
    tracker_file = Path(tracker_path)

    if tracker_file.exists():
        old_df = pd.read_csv(tracker_file)
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_csv(tracker_file, index=False)
    return tracker_file.as_posix(), new_df


def build_dependent_pipeline_example(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_chunks,
    client,
):
    chunks, delayed_chunks = build_dask_chunk_graph(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_chunks,
    )

    @delayed
    def chunk_mean(chunk):
        return float(np.mean(chunk))

    @delayed
    def aggregate_means(values):
        return float(np.mean(values))

    mean_tasks = [chunk_mean(chunk_task) for chunk_task in delayed_chunks]
    total_mean_task = aggregate_means(mean_tasks)
    dependent_result = client.compute(total_mean_task).result()

    return {
        "chunk_count": len(chunks),
        "dependent_task_count": len(mean_tasks) + 1,
        "mean_iteration_value": dependent_result,
    }


def plot_dask_chunk_performance(chunk_rows, output_path="lecture6_dask_chunks.png"):
    x = [row["n_chunks"] for row in chunk_rows]
    time_y = [row["time_s"] for row in chunk_rows]
    speedup_y = [row["speedup"] for row in chunk_rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, time_y, marker="o", color="tab:blue", label="time (s)")
    ax1.set_xlabel("Number of chunks")
    ax1.set_ylabel("Median time (s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, speedup_y, marker="s", color="tab:orange", label="speedup")
    ax2.set_ylabel("Speedup vs 1-worker Dask", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Dask chunk granularity vs performance")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_lecture6_study(
    xmin=-2.0,
    xmax=1.0,
    ymin=-1.5,
    ymax=1.5,
    width=2000,
    height=1500,
    max_iter=500,
    n_workers=4,
    runs=3,
    comparison_width=256,
    comparison_height=256,
    comparison_max_iter=120,
    comparison_runs=1,
):
    n_workers = max(1, min(int(n_workers), os.cpu_count() or 1))
    default_chunks = max(1, 4 * n_workers)

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

    m1 = milestone1_dask_local(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_workers=n_workers,
        n_chunks=default_chunks,
        runs=runs,
    )

    chunk_rows, best_time_row, best_lif_row = benchmark_dask_chunk_sweep(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_workers=n_workers,
        multipliers=(1, 2, 4, 8, 16),
        runs=runs,
    )

    best_chunks = int(best_time_row["n_chunks"])
    comparison_rows = benchmark_full_method_comparison(
        xmin,
        xmax,
        ymin,
        ymax,
        comparison_width,
        comparison_height,
        comparison_max_iter,
        mp_workers=n_workers,
        mp_chunks=max(best_chunks, n_workers),
        dask_workers=n_workers,
        dask_chunks=best_chunks,
        runs=comparison_runs,
    )

    tracker_path, tracker_new_rows = append_performance_tracker(
        comparison_rows,
        width=comparison_width,
        height=comparison_height,
        max_iter=comparison_max_iter,
    )

    mp_time = next(
        row["time_s"]
        for row in comparison_rows
        if row["method"] == "Numba + multiprocessing"
    )
    dask_time = next(
        row["time_s"]
        for row in comparison_rows
        if row["method"] == "Dask (LocalCluster)"
    )

    cluster, client = open_local_cluster(n_workers=n_workers, threads_per_worker=1)
    try:
        e3_pipeline = build_dependent_pipeline_example(
            xmin,
            xmax,
            ymin,
            ymax,
            width=comparison_width,
            height=comparison_height,
            max_iter=comparison_max_iter,
            n_chunks=best_chunks,
            client=client,
        )
        e2_dashboard = client.dashboard_link
    finally:
        client.close()
        cluster.close()

    return {
        "m1": m1,
        "chunk_rows": chunk_rows,
        "best_chunk_time_row": best_time_row,
        "best_chunk_lif_row": best_lif_row,
        "comparison_rows": comparison_rows,
        "tracker_path": tracker_path,
        "tracker_new_rows": tracker_new_rows.to_dict(orient="records"),
        "overhead_vs_multiprocessing": {
            "mp_time_s": mp_time,
            "dask_time_s": dask_time,
            "dask_vs_mp_ratio": dask_time / mp_time,
        },
        "exercise_e2_dashboard_link": e2_dashboard,
        "exercise_e3_pipeline": e3_pipeline,
    }


def to_dataframes(results):
    return (
        pd.DataFrame(results["chunk_rows"]),
        pd.DataFrame(results["comparison_rows"]),
        pd.DataFrame(results["tracker_new_rows"]),
    )
