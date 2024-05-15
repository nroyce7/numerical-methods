import numpy as np


def euler(f, y_0, a, b, n):
    '''
    Uses euler's method to approximate IVPs
    Parameters
    f: differential equation (function)
    y_0: initial values of the dependent variable(s) (float)
    a: initial value of independent variable (float)
    b: final value of indpendent variable (float)

    returns array of t values and y values
    '''
    h = (b-a)/n # step size
    ty = np.zeros((n+1, 2))
    t = a
    y = y_0
    ty[0] = [t, y]
    for i in range(n):
        y = y + h*f(t,y)
        t += h
        ty[i+1] = [t,y]

    return ty


def rk4(f, y_0, a, b, n):
    h = (b-a)/n # step size
    ty = np.zeros((n+1, 2))
    t = a
    y = y_0
    ty[0] = [t, y]

    for i in range(n):
        k_1 = f(t, y)
        k_2 = f(t + (h/2), y + (h*(k_1/2)))
        k_3 = f(t + (h/2), y + (h*(k_2/2)))
        k_4 = f(t + h, y + (h*k_3))

        y = y + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
        t += h
        ty[i+1] = [t,y]

    return ty
