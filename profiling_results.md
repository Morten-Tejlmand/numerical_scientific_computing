# Profiling Results

## naive_profile.prof

```text
Tue Mar  3 16:56:37 2026    naive_profile.prof

         5482039 function calls (5482035 primitive calls) in 1.177 seconds

   Ordered by: cumulative time
   List reduced from 136 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    0.000    0.000    1.065    0.532 C:\Users\morte\AppData\Local\Programs\Python\Python312\Lib\asyncio\base_events.py:1922(_run_once)
        1    0.000    0.000    1.033    1.033 {built-in method builtins.exec}
        1    0.000    0.000    1.033    1.033 <string>:1(<module>)
        1    0.648    0.648    1.033    1.033 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\mandelbrot.py:19(mandelbrot_naive)
  5481530    0.428    0.000    0.428    0.000 {built-in method builtins.abs}
       13    0.075    0.006    0.111    0.009 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\zmq\sugar\socket.py:623(send)
        1    0.000    0.000    0.062    0.062 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\decorator.py:232(fun)
        1    0.000    0.000    0.062    0.062 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\IPython\core\history.py:94(only_when_enabled)
        1    0.000    0.000    0.062    0.062 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\IPython\core\history.py:1047(writeout_cache)
        1    0.014    0.014    0.062    0.062 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\IPython\core\history.py:1031(_writeout_input_cache)


```

## numpy_profile.prof

```text
Tue Mar  3 16:56:37 2026    numpy_profile.prof

         76 function calls in 0.192 seconds

   Ordered by: cumulative time
   List reduced from 41 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.192    0.192 {built-in method builtins.exec}
        1    0.001    0.001    0.191    0.191 <string>:1(<module>)
        1    0.188    0.188    0.190    0.190 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\mandelbrot.py:57(compute_mandelbrot_numpy)
        1    0.000    0.000    0.001    0.001 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:5050(meshgrid)
        1    0.001    0.001    0.001    0.001 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numpy\_core\numeric.py:97(zeros_like)
        3    0.000    0.000    0.001    0.000 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numpy\lib\_function_base_impl.py:5196(<genexpr>)
        2    0.001    0.000    0.001    0.000 {method 'copy' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.000    0.000 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numpy\_core\numeric.py:170(ones)
        2    0.000    0.000    0.000    0.000 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numpy\_core\function_base.py:27(linspace)


```

## hybrid_profile.prof

```text
Tue Mar  3 16:56:40 2026    hybrid_profile.prof

         1670639 function calls (1600372 primitive calls) in 2.622 seconds

   Ordered by: cumulative time
   List reduced from 2849 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     71/1    0.001    0.000    2.636    2.636 {built-in method builtins.exec}
        1    0.000    0.000    2.623    2.623 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numba\core\dispatcher.py:344(_compile_for_args)
     29/1    0.000    0.000    2.615    2.615 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numba\core\dispatcher.py:862(compile)
      8/1    0.000    0.000    2.615    2.615 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numba\core\dispatcher.py:79(compile)
      8/1    0.000    0.000    2.615    2.615 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numba\core\dispatcher.py:86(_compile_cached)
      8/1    0.000    0.000    2.615    2.615 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numba\core\dispatcher.py:101(_compile_core)
      8/1    0.000    0.000    2.615    2.615 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numba\core\compiler.py:713(compile_extra)
        9    0.000    0.000    1.585    0.176 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numba\core\compiler.py:391(__init__)
   280/34    0.000    0.000    1.495    0.044 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numba\core\base.py:261(refresh)
      280    0.003    0.000    1.461    0.005 c:\Users\morte\OneDrive\Desktop\numerical_scientific_computing\.venv\Lib\site-packages\numba\core\cpu.py:97(load_additional_registries)


```

