import time

import numpy as np

from mandelbrot import (
    bench,
    compute_mandelbrot_numpy,
    compute_sums_columns,
    compute_sums_row,
    mandelbrot_naive,
)


# run lecture 1 code in sequence
def mandelbrot_naive_main():
    start = time.time()
    # Compute 1024x1024 grid with 100 max iterations
    arr = mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100)
    elapsed = time.time() - start
    print(f"computation took {elapsed:.3f} seconds")

    np.save("mandelbrot_iters.npy", arr)
    print("saved to mandelbrot_iters.npy")


# lecture 2 mandelbrot optmized big version
def mandelbrot_v2_test():
    start = time.time()
    # Compute 1024x1024 grid with 100 max iterations
    arr = compute_mandelbrot_numpy(-2, 1, -1.5, 1.5, 8192, 8192, 100)
    elapsed = time.time() - start
    print(f"computation took {elapsed:.3f} seconds")

    np.save("mandelbrot_iters.npy", arr)
    print("saved to mandelbrot_iters.npy")


# this takes 63.95 secons with size 8192x8192 and 100 iterations, which is about 4 times faster than the naive version for the same size and iterations
mandelbrot_v2_test()


# trying different scales
sizes = [256, 512, 1024, 2048, 4096]

for n in sizes:
    t_med, t_min, t_max = bench(
        compute_mandelbrot_numpy, -2, 1, -1.5, 1.5, n, n, 100, False
    )
    print(f"{n}x{n}: median={t_med:.4f}s (min={t_min:.4f}, max={t_max:.4f})")

# 256x256: median=0.0206s (min=0.0203, max=0.0250)
# 512x512: median=0.1583s (min=0.1495, max=0.1634)
# 1024x1024: median=0.8980s (min=0.8903, max=0.9064)
# 2048x2048: median=3.6821s (min=3.6316, max=3.7102)
# 4096x4096: median=15.5680s (min=15.0507, max=15.9549)


N = 5000
A = np.random.rand(N, N)
AF = np.asfortranarray(A)

tests = [
    ("Normal row sum", compute_sums_row, A, N),
    ("Normal column sum", compute_sums_columns, A, N),
    ("row_sum asfortranarray", compute_sums_row, AF, N),
    ("col_sum asfortranarray", compute_sums_columns, AF, N),
]

for name, fn, arr, n in tests:
    t_med, t_min, t_max = bench(fn, arr, n, warmup=False)
    print(f"{name}: median={t_med:.4f}s (min={t_min:.4f}, max={t_max:.4f})")
    time.sleep(2)

# bench tested 5 times each with warmup
# Normal row sum: median=0.0297s (min=0.0269, max=0.0310)
# Normal column sum: median=0.1665s (min=0.1619, max=0.1712)
# row sum using asfortranarray: median=0.1750s (min=0.1652, max=0.1797)
# col sum using asfortranarray: median=0.0339s (min=0.0325, max=0.0344)

# bench tested 5 times each without warmup
# Normal row sum: median=0.0318s (min=0.0285, max=0.0602)
# Normal column sum: median=0.1890s (min=0.1649, max=0.1963)
# row sum using asfortranarray: median=0.1724s (min=0.1661, max=0.1866)
# col sum using asfortranarray: median=0.0309s (min=0.0298, max=0.0458)


####
## lecture 3
####

# milestone 1: function level profiling
# cProfile.run(
#     "mandelbrot_naive ( -2 , 1, -1.5 , 1.5 , 512 , 512, 100)", "naive_profile.prof"
# )
# cProfile.run(
#     "compute_mandelbrot_numpy ( -2 , 1, -1.5 , 1.5 , 512 , 512, 100)", "numpy_profile.prof"
# )
# cProfile.run(
#     "compute_mandelbrot_naive_numba ( -2 , 1, -1.5 , 1.5 , 512 , 512, 100)", "hybrid_profile.prof"
# )
# with open("profiling_results.md", "w") as f:
#     f.write("# Profiling Results\n\n")

#     for name in (
#         "naive_profile.prof",
#         "numpy_profile.prof",
#         "hybrid_profile.prof",
#     ):
#         f.write(f"## {name}\n\n")
#         f.write("```text\n")

#         stream = io.StringIO()
#         stats = pstats.Stats(name, stream=stream)
#         stats.sort_stats("cumulative")
#         stats.print_stats(10)

#         f.write(stream.getvalue())
#         f.write("```\n\n")
