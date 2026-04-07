import numpy as np
import matplotlib.pyplot as plt

def exact_sol(u):
    return np.exp(u**2) - 0.5*u**2 - 0.5

def picard_0(u): return 0.5 + 0*u
def picard_1(u): return 0.5 + 0.5*u**2 + 0.25*u**4
def picard_2(u): return 0.5 + 0.5*u**2 + 0.5*u**4 + (1/12)*u**6
def picard_3(u): return 0.5 + 0.5*u**2 + 0.5*u**4 + (1/6)*u**6 + (1/48)*u**8

u_vals = np.linspace(-1, 1, 100)

x_exact = exact_sol(u_vals)
x0 = picard_0(u_vals)
x1 = picard_1(u_vals)
x2 = picard_2(u_vals)
x3 = picard_3(u_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_exact, u_vals, 'k', lw=3, label='Точное решение', alpha=0.3)
plt.plot(x0, u_vals, '--', label='Итерация 0 (x0=0.5)')
plt.plot(x1, u_vals, '--', label='Итерация 1')
plt.plot(x2, u_vals, '--', label='Итерация 2')
plt.plot(x3, u_vals, '--', label='Итерация 3')

plt.title('Сходимость итераций Пикара к точному решению')
plt.xlabel('x')
plt.ylabel('u')
plt.axvline(0, color='black', lw=1)
plt.axhline(0, color='black', lw=1)
plt.grid(True)
plt.legend()
plt.savefig("result2.png", dpi=300)