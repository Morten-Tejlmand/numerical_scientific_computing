import time

import numpy as np


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

    return saved_iter


def mandelbrot_naive_main():
    start = time.time()
    # Compute 1024x1024 grid with 100 max iterations
    arr = mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100)
    elapsed = time.time() - start
    print(f"computation took {elapsed:.3f} seconds")

    np.save("mandelbrot_iters.npy", arr)
    print("saved to mandelbrot_iters.npy")
