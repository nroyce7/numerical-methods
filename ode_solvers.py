import numpy as np
from root_finding import *


def euler(f, y_0, a, b, n):
    """
    Uses euler's method to approximate IVPs

    Parameters
    f: differential equation (function) dy/dt = f(t,y)
    y_0: initial values of the dependent variable(s) (float) y(a) = y_0
    a: initial value of independent variable (float)
    b: final value of independent variable (float)

    returns array of t values and y values

    """
    h = (b-a)/n                 # calculate step size from number of points to avoid casting nonsense
    ty = np.zeros((n+1, 2))     # holds the independent and dependent variables
    t = a
    y = y_0
    ty[0] = [t, y]
    for i in range(n):
        y = y + h*f(t,y)        # follow the tangent line
        t += h
        ty[i+1] = [t,y]

    return ty


def backward_euler(f, y_0, a, b, n):
    h = (b-a)/n                 # calculate step size from number of points to avoid casting nonsense
    ty = np.zeros((n+1, 2))     # holds the independent and dependent variables
    t = a
    y = y_0
    ty[0] = [t, y]

    for i in range(n):
        y_e = y + h*f(t, y)  # follow the tangent line
        t += h
        y = y + h*f(t, y_e)
        ty[i + 1] = [t, y]

    return ty


def rk4(f, y_0, a, b, n):
    """
    Uses Runge-Kutta (4th Order) to approximate IVPs

    Parameters
    f: differential equation (function) dy/dt = f(t,y)
    y_0: initial values of the dependent variable(s) (float)  y(a) = y_0
    a: initial value of independent variable (float)
    b: final value of independent variable (float)

    returns array of t values and y values

    """

    h = (b-a)/n                 # calculate step size from number of points to avoid casting nonsense
    ty = np.zeros((n+1, 2))     # holds the independent and dependent variables
    t = a
    y = y_0
    ty[0] = [t, y]

    for i in range(n):          # Runge-Kutta takes a weighted average of four slopes within each calculation interval
        k_1 = f(t, y)
        k_2 = f(t + (h/2), y + (h*(k_1/2)))
        k_3 = f(t + (h/2), y + (h*(k_2/2)))
        k_4 = f(t + h, y + (h*k_3))

        y = y + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)   # follow the (adjusted) tangent line
        t += h
        ty[i+1] = [t,y]

    return ty
