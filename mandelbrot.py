import statistics
import time

import matplotlib.pyplot as plt
import numpy as np


def viz():
    arr = np.load("mandelbrot.npy")
    plt.imshow(arr, cmap="hot", origin="lower")
    plt.title("Mandelbrot Set")
    plt.colorbar()
    plt.savefig("mandelbrot.png", dpi=200, bbox_inches="tight")
    plt.show()


# lecture 1 naive algorithm for mandelbrot set
def mandelbrot_naive(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iter: int,
    threshold=2.0,
):
    # create a grid of numbers corresponding to the pixel coordinates
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)

    saved_iter = np.empty((height, width), dtype=np.int32)

    # loop over all pixel coordinate
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            c = complex(x, y)
            z = 0.0 + 0.0j
            it = max_iter
            for k in range(max_iter):
                z = z * z + c
                # escape condition
                if abs(z) > threshold:
                    it = k
                    break
            saved_iter[j, i] = it
    np.save("mandelbrot.npy", saved_iter)
    viz()

    return saved_iter


# lecture 2 vectorized algorithm for mandelbrot set
def compute_mandelbrotv2(xmin, xmax, ymin, ymax, width, height, max_iter, show=False):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    # using numpy broadcasting to create a grid of numbers
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros_like(C)
    mask = np.ones(C.shape, dtype=bool)
    # iterate until max_iter or escape
    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + C[mask]
        mask = mask & (np.abs(z) <= 2)
    if show:
        np.save("mandelbrot.npy", mask)
        viz()
    return mask


# benchmarking utility
def bench(fn, *args, runs=5, warmup=True):
    if warmup:
        fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


# trying different scales
sizes = [256, 512, 1024, 2048, 4096]

for n in sizes:
    t_med, t_min, t_max = bench(
        compute_mandelbrotv2, -2, 1, -1.5, 1.5, n, n, 100, False
    )
    print(f"{n}x{n}: median={t_med:.4f}s (min={t_min:.4f}, max={t_max:.4f})")

# 256x256: median=0.0206s (min=0.0203, max=0.0250)
# 512x512: median=0.1583s (min=0.1495, max=0.1634)
# 1024x1024: median=0.8980s (min=0.8903, max=0.9064)
# 2048x2048: median=3.6821s (min=3.6316, max=3.7102)
# 4096x4096: median=15.5680s (min=15.0507, max=15.9549)
# trying one bigger

N = 5000
A = np.random.rand(N, N)


# normal row_sum
def compute_sums_row(A):
    for i in range(N):
        s = np.sum(A[i, :])


# normal column_sum
def compute_sums_columns(A):
    for j in range(N):
        s = np.sum(A[:, j])


AF = np.asfortranarray(A)

tests = [
    ("Normal row sum", compute_sums_row, A),
    ("Normal column sum", compute_sums_columns, A),
    ("row_sum asfortranarray", compute_sums_row, AF),
    ("col_sum asfortranarray", compute_sums_columns, AF),
]

for name, fn, arr in tests:
    t_med, t_min, t_max = bench(fn, arr, warmup=False)
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
