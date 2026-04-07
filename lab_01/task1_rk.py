import numpy as np
import matplotlib.pyplot as plt

a0, a1, a2, a3 = 1, -0.5, 1/24, -1/720

def u_series(x):
    return a0 + a1*x + a2*x**2 + a3*x**3

def solve_euler(x_start, x_end, h, u0, v0):
    steps = int(abs(x_end - x_start) / h)
    actual_h = h if x_end > x_start else -h
    
    x_vals = [x_start]
    u_vals = [u0]
    v_vals = [v0]
    
    for i in range(steps):
        x = x_vals[-1]
        u = u_vals[-1]
        v = v_vals[-1]
        if abs(x) < 1e-10: 
            dv = 1/12
        else:
            dv = -(2*v + u) / (4*x)
        
        v_next = v + actual_h * dv
        u_next = u + actual_h * v
        
        x_vals.append(x + actual_h)
        u_vals.append(u_next)
        v_vals.append(v_next)
        
    return np.array(x_vals), np.array(u_vals)

def solve_rk4(x_start, x_end, h, u0, v0):
    steps = int(abs(x_end - x_start) / h)
    actual_h = h if x_end > x_start else -h
    
    x_vals = [x_start]
    u_vals = [u0]
    v_vals = [v0]
    
    def f_v(x, u, v):
        if abs(x) < 1e-10:
            return 1/12
        return -(2*v + u) / (4*x)
    
    def f_u(x, u, v):
        return v
    
    for i in range(steps):
        x = x_vals[-1]
        u = u_vals[-1]
        v = v_vals[-1]
        
        k1_u = actual_h * f_u(x, u, v)
        k1_v = actual_h * f_v(x, u, v)
        
        k2_u = actual_h * f_u(x + actual_h/2, u + k1_u/2, v + k1_v/2)
        k2_v = actual_h * f_v(x + actual_h/2, u + k1_u/2, v + k1_v/2)
        
        k3_u = actual_h * f_u(x + actual_h/2, u + k2_u/2, v + k2_v/2)
        k3_v = actual_h * f_v(x + actual_h/2, u + k2_u/2, v + k2_v/2)
        
        k4_u = actual_h * f_u(x + actual_h, u + k3_u, v + k3_v)
        k4_v = actual_h * f_v(x + actual_h, u + k3_u, v + k3_v)
        
        u_next = u + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
        v_next = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        
        x_vals.append(x + actual_h)
        u_vals.append(u_next)
        v_vals.append(v_next)
        
    return np.array(x_vals), np.array(u_vals)

h = 0.5
x_right, u_right = solve_euler(0, 10, h, 1, -0.5)
x_left, u_left = solve_euler(0, -10, h, 1, -0.5)

x_full = np.concatenate([x_left[::-1], x_right[1:]])
u_full = np.concatenate([u_left[::-1], u_right[1:]])

x_right_rk4, u_right_rk4 = solve_rk4(0, 10, h, 1, -0.5)
x_left_rk4, u_left_rk4 = solve_rk4(0, -10, h, 1, -0.5)

x_full_rk4 = np.concatenate([x_left_rk4[::-1], x_right_rk4[1:]])
u_full_rk4 = np.concatenate([u_left_rk4[::-1], u_right_rk4[1:]])

plt.figure(figsize=(10, 6))
plt.plot(x_full, u_full, 'b-', label='Метод Эйлера')
plt.plot(x_full_rk4, u_full_rk4, 'g-', label='Метод Рунге-Кутта 4')
plt.plot(x_full, u_series(x_full), 'r--', label='Ряд (4 члена)')
plt.axvline(0, color='black', lw=1)
plt.axhline(0, color='black', lw=0.5)
plt.title('Решение в положительной и отрицательной областях')
plt.legend()
plt.grid(True)
plt.savefig("result1.png", dpi=300)