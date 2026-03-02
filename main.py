import time

import numpy as np

from mandelbrot import compute_mandelbrotv2, mandelbrot_naive


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
    arr = compute_mandelbrotv2(-2, 1, -1.5, 1.5, 8192, 8192, 100)
    elapsed = time.time() - start
    print(f"computation took {elapsed:.3f} seconds")

    np.save("mandelbrot_iters.npy", arr)
    print("saved to mandelbrot_iters.npy")


# this takes 63.95 secons
mandelbrot_v2_test()
