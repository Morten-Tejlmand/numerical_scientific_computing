import logging
import os
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask import delayed
from dask.distributed import Client, LocalCluster

from mandelbrot import compute_mandelbrot_naive_numba
from mandelprot_lecture_5 import compute_mandelbrot_chunk_numba, split_rows_into_chunks


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


def compute_mandelbrot_dask(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    n_chunks,
    client,
    worker_addresses=None,
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

    if worker_addresses:
        futures = client.compute(
            delayed_chunks,
            workers=list(worker_addresses),
            allow_other_workers=False,
        )
    else:
        futures = client.compute(delayed_chunks)

    chunk_results = client.gather(futures)
    return np.vstack(chunk_results)


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
    worker_addresses=None,
):
    compute_mandelbrot_dask(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        n_chunks,
        client,
        worker_addresses=worker_addresses,
    )

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        compute_mandelbrot_dask(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_chunks,
            client,
            worker_addresses=worker_addresses,
        )
        times.append(time.perf_counter() - t0)

    return statistics.median(times), min(times), max(times)


def open_local_cluster(
    n_workers=3,
    threads_per_worker=1,
    scheduler_port=8786,
    dashboard_port=8787,
):
    logging.getLogger("distributed").setLevel(logging.ERROR)
    logging.getLogger("dask").setLevel(logging.ERROR)

    scheduler_port = int(scheduler_port) if scheduler_port is not None else 0
    if dashboard_port in (None, 0):
        dashboard_address = ":0"
    else:
        dashboard_address = f":{int(dashboard_port)}"

    cluster = LocalCluster(
        n_workers=int(n_workers),
        threads_per_worker=int(threads_per_worker),
        processes=True,
        scheduler_port=scheduler_port,
        dashboard_address=dashboard_address,
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    return cluster, client


def connect_to_cluster(scheduler_address, timeout_seconds=30):
    client = Client(scheduler_address, timeout=f"{int(timeout_seconds)}s")
    client.wait_for_workers(1, timeout=timeout_seconds)
    return client


def verify_cluster_versions(client):
    def _worker_versions():
        import dask
        import numba
        import numpy as np
        import sys

        return {
            "python": sys.version.split(" ")[0],
            "dask": dask.__version__,
            "numpy": np.__version__,
            "numba": numba.__version__,
        }

    versions_by_worker = client.run(_worker_versions)

    signature = None
    matching_versions = True
    for version_info in versions_by_worker.values():
        current = (
            version_info["python"],
            version_info["dask"],
            version_info["numpy"],
            version_info["numba"],
        )
        if signature is None:
            signature = current
            continue
        if current != signature:
            matching_versions = False
            break

    return {
        "matching_versions": bool(matching_versions),
        "versions_by_worker": versions_by_worker,
    }


def benchmark_numba_baseline(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    runs=3,
):
    median_s, min_s, max_s = bench(
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
    return {"median_s": median_s, "min_s": min_s, "max_s": max_s}


def benchmark_chunk_sweep(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    client,
    runs=3,
    chunk_multipliers=(1, 2, 4, 8, 16),
    comparison_local_workers=None,
):
    worker_addresses = sorted(client.scheduler_info()["workers"].keys())
    n_workers = len(worker_addresses)
    if n_workers <= 0:
        raise RuntimeError("No workers available in Dask cluster.")

    chunk_candidates = sorted(
        {
            max(1, min(int(height), int(multiplier) * n_workers))
            for multiplier in chunk_multipliers
        }
    )

    rows = []
    for n_chunks in chunk_candidates:
        median_s, min_s, max_s = benchmark_dask_config(
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
            worker_addresses=worker_addresses,
        )
        rows.append(
            {
                "n_chunks": n_chunks,
                "time_s": median_s,
                "min_s": min_s,
                "max_s": max_s,
            }
        )

    best_row = min(rows, key=lambda x: x["time_s"])
    best_chunks = int(best_row["n_chunks"])

    if comparison_local_workers is None:
        comparison_local_workers = min(os.cpu_count() or 1, max(1, n_workers))
    comparison_local_workers = max(1, int(comparison_local_workers))

    local_cluster, local_client = open_local_cluster(
        n_workers=comparison_local_workers,
        threads_per_worker=1,
        scheduler_port=0,
        dashboard_port=0,
    )
    try:
        local_median_s, _, _ = benchmark_dask_config(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_chunks=best_chunks,
            client=local_client,
            runs=runs,
        )
    finally:
        local_client.close()
        local_cluster.close()

    local_comparison = {
        "local_workers": comparison_local_workers,
        "best_chunks": best_chunks,
        "remote_time_s": float(best_row["time_s"]),
        "local_time_s": float(local_median_s),
        "remote_vs_local_ratio": float(best_row["time_s"] / local_median_s),
    }

    return rows, best_row, local_comparison


def benchmark_worker_scaling(
    xmin,
    xmax,
    ymin,
    ymax,
    width,
    height,
    max_iter,
    best_chunks,
    client,
    worker_counts=None,
    runs=3,
):
    available_workers = sorted(client.scheduler_info()["workers"].keys())
    max_workers = len(available_workers)
    if max_workers <= 0:
        raise RuntimeError("No workers available in Dask cluster.")

    if worker_counts is None:
        worker_counts = list(range(1, max_workers + 1))
    else:
        worker_counts = sorted(
            {
                int(w)
                for w in worker_counts
                if int(w) >= 1 and int(w) <= max_workers
            }
        )

    if not worker_counts:
        worker_counts = [1]
    if 1 not in worker_counts:
        worker_counts = [1] + worker_counts

    rows = []
    for worker_count in worker_counts:
        subset = available_workers[:worker_count]
        median_s, min_s, max_s = benchmark_dask_config(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            n_chunks=best_chunks,
            client=client,
            runs=runs,
            worker_addresses=subset,
        )
        rows.append(
            {
                "workers": worker_count,
                "time_s": median_s,
                "min_s": min_s,
                "max_s": max_s,
            }
        )

    t1_s = next(row["time_s"] for row in rows if row["workers"] == 1)
    for row in rows:
        row["speedup_vs_1worker"] = t1_s / row["time_s"]
        row["ideal_speedup"] = float(row["workers"])

    return rows


def get_cluster_config(client):
    info = client.scheduler_info()
    workers = info.get("workers", {})

    thread_counts = [workers[address].get("nthreads", 0) for address in workers]
    cpus_per_vm = max(thread_counts) if thread_counts else 0

    return {
        "scheduler_address": info.get("address"),
        "dashboard_link": client.dashboard_link,
        "n_workers": len(workers),
        "cpus_per_vm_estimate": int(cpus_per_vm),
    }


def plot_chunk_sweep(chunk_rows, output_path="lecture7_chunk_sweep.png"):
    x = [row["n_chunks"] for row in chunk_rows]
    y = [row["time_s"] for row in chunk_rows]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("Number of chunks")
    plt.ylabel("Median wall time (s)")
    plt.title("Lecture 7: chunk sweep on Dask cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_worker_scaling(worker_rows, output_path="lecture7_worker_scaling.png"):
    x = [row["workers"] for row in worker_rows]
    y_time = [row["time_s"] for row in worker_rows]
    y_speedup = [row["speedup_vs_1worker"] for row in worker_rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, y_time, marker="o", color="tab:blue")
    ax1.set_xlabel("Number of workers")
    ax1.set_ylabel("Median wall time (s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, y_speedup, marker="s", color="tab:orange", label="speedup")
    ax2.set_ylabel("Speedup vs 1 worker", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Lecture 7: worker scaling (fixed best chunk size)")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_lecture7_study(
    scheduler_address="tcp://127.0.0.1:8786",
    xmin=-2.0,
    xmax=1.0,
    ymin=-1.5,
    ymax=1.5,
    width=4096,
    height=4096,
    max_iter=200,
    runs=3,
    chunk_multipliers=(1, 2, 4, 8, 16),
    worker_counts=None,
    launch_local_if_unreachable=True,
    local_emulation_workers=3,
    local_emulation_threads=1,
    local_emulation_scheduler_port=8786,
    local_emulation_dashboard_port=8787,
    comparison_local_workers=None,
):
    if int(width) < 4096 or int(height) < 4096:
        raise ValueError("Lecture 7 study requires N >= 4096 for both width and height.")

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

    cluster = None
    client = None
    cluster_mode = "remote"
    connect_error = None

    try:
        try:
            client = connect_to_cluster(scheduler_address)
        except Exception as exc:
            connect_error = str(exc)
            if not launch_local_if_unreachable:
                raise

            cluster_mode = "local_emulation"
            cluster, client = open_local_cluster(
                n_workers=local_emulation_workers,
                threads_per_worker=local_emulation_threads,
                scheduler_port=local_emulation_scheduler_port,
                dashboard_port=local_emulation_dashboard_port,
            )
            scheduler_address = cluster.scheduler_address

        version_check = verify_cluster_versions(client)
        cluster_config = get_cluster_config(client)
        numba_baseline = benchmark_numba_baseline(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            runs=runs,
        )

        chunk_rows, best_chunk_row, local_compare = benchmark_chunk_sweep(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            client=client,
            runs=runs,
            chunk_multipliers=chunk_multipliers,
            comparison_local_workers=comparison_local_workers,
        )
        best_chunks = int(best_chunk_row["n_chunks"])

        worker_rows = benchmark_worker_scaling(
            xmin,
            xmax,
            ymin,
            ymax,
            width,
            height,
            max_iter,
            best_chunks=best_chunks,
            client=client,
            worker_counts=worker_counts,
            runs=runs,
        )
        best_scaling_row = min(worker_rows, key=lambda x: x["time_s"])
        speedup_vs_numba = numba_baseline["median_s"] / best_scaling_row["time_s"]

        return {
            "cluster_mode": cluster_mode,
            "connect_error": connect_error,
            "scheduler_address": scheduler_address,
            "dashboard_link": client.dashboard_link,
            "version_check": version_check,
            "cluster_config": cluster_config,
            "grid": {
                "width": int(width),
                "height": int(height),
                "max_iter": int(max_iter),
            },
            "numba_baseline": numba_baseline,
            "chunk_rows": chunk_rows,
            "best_chunk_row": best_chunk_row,
            "local_dask_comparison": local_compare,
            "worker_scaling_rows": worker_rows,
            "best_worker_row": best_scaling_row,
            "speedup_vs_numba_baseline": speedup_vs_numba,
        }
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()


def to_dataframes(results):
    return (
        pd.DataFrame(results["chunk_rows"]),
        pd.DataFrame(results["worker_scaling_rows"]),
    )
