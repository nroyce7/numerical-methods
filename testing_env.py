from ode_solvers import *
from root_finding import *
import matplotlib.pyplot as plt


def f(t, y):
    return np.sin(10*t)


t = np.linspace(0, np.pi, 1000)
n = 100
sol_e = euler(f, 1, 0, np.pi, n)
sol_eb = backward_euler(f, 1, 0, np.pi, n)

plt.plot(sol_e[:,0], sol_e[:,1], sol_eb[:,0], sol_eb[:,1])
plt.plot(t,-1/10*np.cos(10*t)+11/10)
plt.legend(['Euler', 'Back Euler', 'Exact'])
plt.show()
'''


def f(x):
    return x**3 - x - 2


def f_prime(x):
    return 3*x**2 - 1


print('\n\n\n\n\n')

print('Newton Method')
print('----------------------------')
x = newton_method(f, f_prime, 2)
print(x)
print('\n')

print('Bisection Method')
print('----------------------------')
x = bisection_method(f, 1, 2)
print(x)
print('\n')

print('False Position')
print('----------------------------')
x = false_position(f, 1, 2)
print(x)
print('\n')

print('Secant Method')
print('----------------------------')
x = secant_method(f, 1, 2)
print(x)
print('\n')
'''