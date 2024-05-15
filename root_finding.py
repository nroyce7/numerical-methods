import numpy as np


def bisection_method(f, a, b, tol = 1e-4, max_itr = 100):
    if f(a)*f(b) > 0:
        print('Root not in interval')
        return -1

    for i in range(max_itr):
        c = (a+b)/2
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c

        if abs(f(c)) < tol:
            print(f'Root within tolerance found after {i} iterations.')
            return c
    print('Maximum iterations reached')
    return c


def false_position(f, a, b, tol = 1e-4, max_itr = 100):
    if f(a)*f(b) > 0:
        print('Root not in interval')
        return -1

    for i in range(max_itr):
        c = (a*f(b) - b*f(a))/(f(b) - f(a))

        if f(a)*f(c) < 0:
            b = c
        else:
            a = c

        if abs(f(c)) < tol:
            print(f'Root within tolerance found after {i} iterations.')
            return c
    print('Maximum iterations reached')
    return c


def newton_method(f, f_prime, x_0, tol = 1e-4, max_itr = 100):
    x = x_0
    for i in range(max_itr):
        x_n = x - (f(x)/f_prime(x))

        if abs(x_n - x) < tol:
            print(f'Root within tolerance found after {i} iterations.')
            return x

        x = x_n
        
    print('Maximum iterations reached')
    return x


def secant_method(f, x_0, x_1, tol = 1e-4, max_itr = 100):
    for i in range(max_itr):
        x_2 = x_1 - f(x_1)*((x_1 - x_0)/(f(x_1) - f(x_0)))
        x_0, x_1 = x_1, x_2

        if abs(f(x_1)) < tol:
            print(f'Root within tolerance found after {i} iterations.')
            return x_1

    print('Maximum iterations reached')
    return x_1
