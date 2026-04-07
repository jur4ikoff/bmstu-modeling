import numpy as np
import matplotlib.pyplot as plt

R = 0.35
l = 12
L_k = 187 * (10**(-6))
C_k = 268 * (10**(-6))
R_k = 0.25
U_co = 1400
I_o = 0.3
T_w = 2000

I_table = [0.5, 1, 5, 10, 50, 200, 400, 800, 1200]
T0_I_table = [6730, 6790, 7150, 7270, 8010, 9185, 10010, 11140, 12010]
m_I_table = [0.50, 0.55, 1.7, 3, 11, 32, 40, 41, 39]
T_sigma_table = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000]
Sigma_table = [0.031, 0.27, 2.05, 6.06, 12.0, 19.9, 29.6, 41.1, 54.1, 67.7, 81.5]

R_k_fix = 0.25

def f(t, I, U):
    return (U - (R_k + R) * I) / L_k

def T_z(T0, z, m):
    return T0 + (T_w - T0) * z**m

def get_sigma_z(z, curr_T0, curr_m):
    T_z = curr_T0 + (T_w - curr_T0) * (z**curr_m)
    sigma = lagrange_interp(T_sigma_table, Sigma_table, T_z)
    return sigma * z


def integrate_simpson(curr_T0, curr_m, steps=100):
    if steps % 2 != 0:
        steps += 1
        
    h = 1.0 / steps
    
    res = get_sigma_z(0, curr_T0, curr_m) + get_sigma_z(1, curr_T0, curr_m)
    
    for i in range(1, steps):
        z = i * h
        if i % 2 != 0:
            res += 4 * get_sigma_z(z, curr_T0, curr_m)
        else:
            res += 2 * get_sigma_z(z, curr_T0, curr_m)
            

    return (h / 3.0) * res

# сопротивление R_p для текущего тока
def calc_rp(current):
    curr_T0 = lagrange_interp(I_table, T0_I_table, current)
    curr_m = lagrange_interp(I_table, m_I_table, current)
    
    integral_val = integrate_simpson(curr_T0, curr_m)
    
    denom = 2 * np.pi * (R**2) * integral_val
    return l / denom, curr_T0


# интерполяция чепрез полином лагранжа
def lagrange_interp(x_table, y_table, x_val):
    locality = 4
    size = len(x_table)
    idx = 0
    # x_table[idx] <= x_val <= x_table[idx + 1]
    while idx < size and x_table[idx] < x_val:
        idx += 1

    half = locality // 2
    left = idx - half
    right = left + locality

    if left < 0:
        left = 0
        right = locality
    if right > size:
        right = size
        left = size - locality

    x_window = x_table[left:right]
    y_window = y_table[left:right]
    
    res = 0
    for i in range(locality):
        term = y_window[i]
        for j in range(locality):
            if i != j:
                term *= (x_val - x_window[j]) / (x_window[i] - x_window[j]) 
        res += term

    return res

def system_odes(I, U, Rp):
    """ Правые части системы ОДУ """
    dI_dt = (U - (R_k + Rp) * I) / L_k
    dU_dt = -I / C_k
    return dI_dt, dU_dt

def multiply_arrs(arr1, arr2):
    res = [0 for _ in range(len(arr1))]
    for i in range(len(arr1)):
        res[i] = arr1[i] * arr2[i]
    return res

def solve_system(mode="plasma", t_end=600e-6):
    t, dt = 0, 1e-7
    curr_I, curr_U = I_o, U_co
    res_t, res_I, res_U, res_Rp, res_T0 = [], [], [], [], []

    while t <= t_end:
        rp, t0 = calc_rp(curr_I)
        
        if mode == "plasma":
            pass
        elif mode == "zero":
            rp = 0
        elif mode == "fixed_200":
            rp = 200

        res_t.append(t * 1e6)
        res_I.append(curr_I)
        res_U.append(curr_U)
        res_Rp.append(rp)
        res_T0.append(t0)

        k1_I, k1_U = system_odes(curr_I, curr_U, rp)
        k2_I, k2_U = system_odes(curr_I + (dt/2)*k1_I, curr_U + (dt/2)*k1_U, rp)
        k3_I, k3_U = system_odes(curr_I + (dt/2)*k2_I, curr_U + (dt/2)*k2_U, rp)
        k4_I, k4_U = system_odes(curr_I + dt*k3_I, curr_U + dt*k3_U, rp)
        
        curr_I += (dt/6) * (k1_I + 2*k2_I + 2*k3_I + k4_I)
        curr_U += (dt/6) * (k1_U + 2*k2_U + 2*k3_U + k4_U)
        t += dt
        
    return np.array(res_t), np.array(res_I), np.array(res_U), np.array(res_Rp), np.array(res_T0)

#defense 
def solve_backward(I_end, U_end, t_start_val, dt):
    t = t_start_val
    curr_I = I_end
    curr_U = U_end
    
    res_t, res_I, res_U = [], [], []

    while t >= -1e-9:
        res_t.append(t * 1e6) 
        res_I.append(curr_I)
        res_U.append(curr_U)

        def get_derivatives(I_val, U_val):
            rp_val, _ = calc_rp(I_val)
            return system_odes(I_val, U_val, rp_val)

        h = -dt
        
        k1_I, k1_U = get_derivatives(curr_I, curr_U)
        k2_I, k2_U = get_derivatives(curr_I + (h/2)*k1_I, curr_U + (h/2)*k1_U)
        k3_I, k3_U = get_derivatives(curr_I + (h/2)*k2_I, curr_U + (h/2)*k2_U)
        k4_I, k4_U = get_derivatives(curr_I + h*k3_I, curr_U + h*k3_U)
        
        curr_I += (h/6) * (k1_I + 2*k2_I + 2*k3_I + k4_I)
        curr_U += (h/6) * (k1_U + 2*k2_U + 2*k3_U + k4_U)
        t += h
        
    return np.array(res_t), np.array(res_I), np.array(res_U)

def print_report(t, I, U, Rp, T0, dt, t_imp):
    i_max_idx = np.argmax(I)
    print("\n" + "="*50)
    print(f"{'СВОДНЫЙ ОТЧЕТ ПО МОДЕЛИРОВАНИЮ':^50}")
    print("="*50)
    print(f"Шаг сетки (dt):            {dt*1e6:<10.3f} мкс")
    print(f"Максимальный ток (I_max):  {I[i_max_idx]:<10.2f} А  (на {t[i_max_idx]:.1f} мкс)")
    print(f"Длительность импульса:     {t_imp:<10.2f} мкс")
    print(f"Напряжение в конце:        {U[-1]:<10.2f} В")
    print(f"Макс. температура (T0):    {np.max(T0):<10.2f} К")
    print(f"Мин. сопротивление Rp:     {np.min(Rp):<10.4f} Ом")
    print(f"Макс. сопротивление Rp:    {np.max(Rp):<10.4f} Ом")
    print("="*50)

    print(f"\n{'ТАБЛИЦА РЕЗУЛЬТАТОВ (ВЫБОРКА)':^60}")
    header = f"{'t, мкс':^10} | {'I, А':^10} | {'U, В':^10} | {'Rp, Ом':^10} | {'T0, К':^10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    indices = np.linspace(0, len(t) - 1, 15, dtype=int)
    
    for idx in indices:
        print(f"{t[idx]:^10.1f} | {I[idx]:^10.2f} | {U[idx]:^10.1f} | {Rp[idx]:^10.3f} | {T0[idx]:^10.0f}")
    print("-" * len(header) + "\n")

def main():
    t_p, I_p, U_p, Rp_p, T0_p = solve_system(mode="plasma")
    dt = 2 * 10**-6
    
    i_max = np.max(I_p)
    level = 0.35 * i_max
    idx_above = np.where(I_p >= level)[0]
    t_s, t_e = t_p[idx_above[0]], t_p[idx_above[-1]]
    t_imp = t_e - t_s
    print_report(t_p, I_p, U_p, Rp_p, T0_p, dt, t_imp)
    fig1, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig1.suptitle(f"Задание №1: Основные характеристики (Шаг сетки dt=0.1 мкс, t_имп={t_imp:.1f} мкс)", fontsize=14)

    # I(t)
    axes[0, 0].plot(t_p, I_p, 'r', label='I(t), А')
    axes[0, 0].axhline(y=level, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].plot([t_s, t_e], [level, level], 'blue', marker='o', markersize=4)
    axes[0, 0].set_title("Ток в контуре")
    axes[0, 0].grid(); axes[0, 0].legend()

    # U(t)
    axes[0, 1].plot(t_p, U_p, 'b', label='U(t), В')
    axes[0, 1].set_title("Напряжение на конденсаторе")
    axes[0, 1].grid(); axes[0, 1].legend()

    # Rp(t)
    axes[1, 0].plot(t_p, Rp_p, 'g', label='Rp(t), Ом')
    axes[1, 0].set_title("Сопротивление плазмы")
    axes[1, 0].grid(); axes[1, 0].legend()

    # I(t) * Rp(t)
    axes[1, 1].plot(t_p, I_p * Rp_p, 'purple', label='I(t)*Rp(t), В')
    axes[1, 1].set_title("Напряжение на плазменной трубке")
    axes[1, 1].grid(); axes[1, 1].legend()

    # T0(t)
    axes[2, 0].plot(t_p, T0_p, 'orange', label='T0(t), K')
    axes[2, 0].set_title("Температура на оси")
    axes[2, 0].set_xlabel("Время, мкс")
    axes[2, 0].grid(); axes[2, 0].legend()
    
    axes[2, 1].axis('off')

    fig1.savefig("task_1-4.svg")
    t_z, I_z, _, _, _ = solve_system(mode="zero", t_end=1500e-6)
    t_f, I_f, _, _, _ = solve_system(mode="fixed_200", t_end=20e-6)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(t_z, I_z, color='brown')
    ax1.set_title("Задание №2: Rk + Rp = 0 (Незатухающие)")
    ax1.set_xlabel("Время, мкс"); ax1.grid()

    ax2.plot(t_f, I_f, color='black')
    ax2.set_title("Задание №3: Rk + Rp = 200 Ом (Апериодический)")
    ax2.set_xlabel("Время, мкс"); ax2.grid()

    plt.tight_layout()
    fig2.savefig("task_2-3.svg")
    
    I_end = I_p[-1]
    U_end = U_p[-1]
    t_end_val = t_p[-1] / 1e6
    
    t_b, I_b, U_b = solve_backward(I_end, U_end, t_end_val, dt)

    fig_rev, (ax_i, ax_u) = plt.subplots(2, 1, figsize=(10, 8))
    fig_rev.suptitle("Тест на обратимость решения (Forward vs Backward)", fontsize=14)

    ax_i.plot(t_p, I_p, color='red', label='Прямой ход (I forward)', linewidth=3, alpha=0.5)
    ax_i.plot(t_b, I_b, color='blue', linestyle='--', label='Обратный ход (I backward)', linewidth=2)
    ax_i.set_ylabel("Ток I, А")
    ax_i.grid(True, linestyle=':')
    ax_i.legend()

    ax_u.plot(t_p, U_p, color='darkgreen', label='Прямой ход (U forward)', linewidth=3, alpha=0.5)
    ax_u.plot(t_b, U_b, color='lime', linestyle='--', label='Обратный ход (U backward)', linewidth=2)
    ax_u.set_ylabel("Напряжение U, В")
    ax_u.set_xlabel("Время, мкс")
    ax_u.grid(True, linestyle=':')
    ax_u.legend()

    plt.tight_layout()
    fig_rev.savefig("reversibility_full.png")
    i_end_back = I_b[-1]
    u_end_back = U_b[-1]
    plt.show()
    print(i_end_back, u_end_back)

if __name__ == "__main__":
    main()