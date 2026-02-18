import matplotlib.pyplot as plt
import numpy as np


def main():
    arr = np.load("mandelbrot_iters.npy")

    plt.imshow(arr, cmap="hot", origin="lower")
    plt.title("Mandelbrot Set")
    plt.colorbar()
    plt.savefig("mandelbrot.png", dpi=200, bbox_inches="tight")
    plt.show()
