import numpy as np

def approx_derivative(f, a, b, n):
    h = (b-a)/n
    diff = np.zeros(n)
    for i in range(n):
        if i == n-1:
            diff[i] = (f(x) - f(x-h))/h
        else:
            diff[i] = (f(x+h) - f(x))/h

    return diff
