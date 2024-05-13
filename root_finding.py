import numpy as np


def newton_method(f, f_prime, x_0, tol = 1e-4, max_itr = 100):
    x = x_0
    for i in range(max_itr):
        x_n = x - (y/y_p)

        if abs(x_n - x) < tol:
            print(f'Root within tolerance found after {i} iterations.')
            return x

        x = x_n
        
    print('Maximum iterations reached')
    return x
