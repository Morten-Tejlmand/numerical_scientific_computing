import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


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
    show=False,
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
    if show:
        np.save("mandelbrot.npy", saved_iter)
        viz()

    return saved_iter


# lecture 2 vectorized algorithm for mandelbrot set
def compute_mandelbrot_numpy(
    xmin, xmax, ymin, ymax, width, height, max_iter, show=False
):
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


# normal row_sum
def compute_sums_row(A, N):
    for i in range(N):
        s = np.sum(A[i, :])


# normal column_sum
def compute_sums_columns(A, N):
    for j in range(N):
        s = np.sum(A[:, j])


#####
## lecture 3
#####


@njit
def compute_mandelbrot_point_numba(c, max_iter=100):
    z = 0j
    for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return max_iter


def compute_mandelbrot_hybrid(
    xmin, xmax, ymin, ymax, width, height, max_iter, display=False
):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = compute_mandelbrot_point_numba(c, max_iter)
    return result


@njit
def compute_mandelbrot_naive_numba(
    xmin, xmax, ymin, ymax, width, height, max_iter, display=False
):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and z.real * z.real + z.imag * z.imag <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n
    return result


@njit
def compute_mandelbrot_numba_float64(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n

    return result


@njit
def compute_mandelbrot_numba_float32(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width).astype(np.float32)
    y = np.linspace(ymin, ymax, height).astype(np.float32)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n

    return result


@njit
def compute_mandelbrot_numba_float16(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width).astype(np.float16)
    y = np.linspace(ymin, ymax, height).astype(np.float16)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n

    return result
