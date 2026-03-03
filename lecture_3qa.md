# lecture 3 qa

## milestone 1

**Using this setup**
cProfile.run(
"mandelbrot_naive ( -2 , 1, -1.5 , 1.5 , 512 , 512, 100)", "naive_profile.prof"
)
cProfile.run(
"compute_mandelbrot_numpy ( -2 , 1, -1.5 , 1.5 , 512 , 512, 100)", "numpy_profile.prof"
)

### Q1: Which function takes most total time?

A1: The naive method takes longer. Naive takkes 0.963 s. , compared too 0.401 s.

### Q2: Are there functions called surprisingly many times?

A2: The escape call inside the naive method, it is being called 5.481.530.

### Q3: How does NumPy profile compare to naive?

It is faster not spending much time with pythohn calls. The naive spend a lot of time doing python level calls such as abs

### Q4: Where does NumPy spend its time?

It spends it´s time in c.
Almost all runtime is in compute_mandelbrot_numpy cumtime: 0.190 s, which delegates work to Numpys compiled array operations.
The only visible Python-level costs are small one-off array constructions like meshgrid 0.001 s and zeros_like 0.001 .

## milestone 2

### cProfile on naive vs NumPy: How many functions appear in each profile? What does this difference tell you about where the work actually happens?

A:
Naive: ~5.5 million inner-loop executions
NumPy: ~100 vectorized operations

### line profiler on naive: Which lines dominate runtime? What fraction of total time is spent in the inner loop?

A: the lines in the inner loop is where the is used most time.

### Based on your profiling results: why is NumPy faster than naive Python?

A numpy does not have multiple nested loops, which it has to call many millions times.

### What would you need to change to make the naive version faster? (hint: what does line profiler tell you about the inner loop?)

A: Not use abs(), pre-allocate the list, thus not using append. because it removes the possibility of re allocating memory for the list

## milestone 3

comparison in profiling_results.md

## milestone 4
