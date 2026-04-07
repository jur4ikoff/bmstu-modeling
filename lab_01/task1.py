import numpy as np
import matplotlib.pyplot as plt

a0, a1, a2, a3 = 1, -0.5, 1/24, -1/720

def u_series(x):
    return a0 + a1*x + a2*x**2 + a3*x**3

def solve_euler(x_start, x_end, h, u0, v0):
    steps = int(abs(x_end - x_start) / h)
    actual_h = h if x_end > x_start else -h
    x_vals, u_vals, v_vals = [float(x_start)], [float(u0)], [float(v0)]
    zero_crossings = []
    
    for _ in range(steps):
        x, u, v = x_vals[-1], u_vals[-1], v_vals[-1]
        
        if abs(x) < 1e-10:
            dv = 1/12
        else:
            dv = -(2*v + u) / (4*x)
            
        v_next = v + actual_h * dv
        u_next = u + actual_h * v
        
        if len(u_vals) > 1:
            if (u_vals[-1] > 0 and u_next <= 0) or (u_vals[-1] < 0 and u_next >= 0):
                zero_crossings.append(x + actual_h)
        
        x_vals.append(x + actual_h)
        u_vals.append(u_next)
        v_vals.append(v_next)
        
    return np.array(x_vals), np.array(u_vals), zero_crossings

h = 0.001

# Меньше первого отброшенного члена

x_right, u_right, zeros_right = solve_euler(0, 20, h, 1, -0.5)
x_left, u_left, zeros_left = solve_euler(0, -5, h, 1, -0.5)

print("Точки пересечения с нулём (метод Эйлера):")
if zeros_right:
    print(f"  Положительная область:")
    for i, z in enumerate(zeros_right, 1):
        print(f"    Ноль #{i}: x ≈ {z:.6f}")
if zeros_left:
    print(f"  Отрицательная область:")
    for i, z in enumerate(zeros_left, 1):
        print(f"    Ноль #{i}: x ≈ {z:.6f}")

x_full = np.concatenate([x_left[::-1], x_right[1:]])
u_full = np.concatenate([u_left[::-1], u_right[1:]])

plt.figure(figsize=(10, 6))
plt.plot(x_full, u_full, 'b-', label='Метод Эйлера', linewidth=2)
plt.plot(x_full, u_series(x_full), 'r--', label='Ряд (4 члена)')

colors = ['green', 'purple', 'brown', 'pink']
if zeros_right:
    for i, z in enumerate(zeros_right):
        color = colors[i % len(colors)]
        plt.axvline(z, color=color, linestyle=':', linewidth=1.5, label=f'Ноль при x={z:.3f}')
if zeros_left:
    for i, z in enumerate(zeros_left):
        color = colors[(i + len(zeros_right)) % len(colors)]
        plt.axvline(z, color=color, linestyle=':', linewidth=1.5, label=f'Ноль при x={z:.3f}')

plt.axvline(0, color='black', lw=1)
plt.axhline(0, color='black', lw=0.5)
plt.title('Решение в положительной и отрицательной областях')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("result1.png", dpi=300)