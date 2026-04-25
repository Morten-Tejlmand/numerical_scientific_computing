import numpy as np
import pytest

from mandelbrot import (
    compute_mandelbrot_naive_numba,
    compute_mandelbrot_numba_float64,
    compute_mandelbrot_point_numba,
)

# Convention for compute_mandelbrot_point_numba: check |z|^2 > 4 *before* update.
# n=0 → check z=0 (passes), update z = c; n=1 → check |c|^2 > 4 → return 1 if true.
POINT_CASES = [
    (0 + 0j, 100, 100),     # origin: z stays 0, never escapes
    (-2.5 + 0j, 100, 1),    # |c|^2 = 6.25 > 4, escapes at n=1
    (0 + 3j, 100, 1),       # |c|^2 = 9 > 4, escapes at n=1
    (2 + 2j, 100, 1),       # |c|^2 = 8 > 4, escapes at n=1
]


@pytest.mark.parametrize("c, max_iter, expected", POINT_CASES)
def test_point_numba_known_values(c: complex, max_iter: int, expected: int) -> None:
    assert compute_mandelbrot_point_numba(c, max_iter) == expected


def test_naive_numba_output_shape() -> None:
    result = compute_mandelbrot_naive_numba(-2.0, 1.0, -1.5, 1.5, 32, 24, 50)
    assert result.shape == (24, 32)


def test_naive_numba_iteration_bounds() -> None:
    max_iter = 50
    result = compute_mandelbrot_naive_numba(-2.0, 1.0, -1.5, 1.5, 32, 32, max_iter)
    assert int(result.min()) >= 0
    assert int(result.max()) <= max_iter


def test_naive_numba_agrees_with_float64() -> None:
    args = (-2.0, 1.0, -1.5, 1.5, 32, 32, 50)
    naive = compute_mandelbrot_naive_numba(*args)
    f64 = compute_mandelbrot_numba_float64(*args)
    np.testing.assert_array_equal(naive, f64)


def test_point_numba_respects_max_iter() -> None:
    # c=0: never escapes; result must equal whatever max_iter we pass
    for max_iter in (10, 50, 200):
        assert compute_mandelbrot_point_numba(0 + 0j, max_iter) == max_iter


def test_naive_numba_interior_point_saturates() -> None:
    # c=-0.5+0j is deep inside the set; all iterations should reach max_iter
    max_iter = 50
    result = compute_mandelbrot_naive_numba(-0.5, -0.5, 0.0, 0.0, 1, 1, max_iter)
    assert int(result[0, 0]) == max_iter
