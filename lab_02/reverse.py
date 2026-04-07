import math
import matplotlib.pyplot as plt

R_radius = 0.35
l_p = 12.0
L_k = 187e-6
C_k = 268e-6
R_k = 0.25
U_0 = 1400
I_0 = 0.5
T_w = 2000.0

table1_I = [0.5, 1, 5, 10, 50, 200, 400, 800, 1200]
table1_T0 = [6730, 6790, 7150, 7270, 8010, 9185, 10010, 11140, 12010]
table1_m = [0.50, 0.55, 1.7, 3, 11, 32, 40, 41, 39]

table2_T = [
    4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000
]
table2_sigma = [
    0.031, 0.27, 2.05, 6.06, 12.0, 19.9, 29.6, 41.1, 54.1, 67.7, 81.5
]


def interpolate(x, x_data, y_data):
    if x <= x_data[0]:
        return y_data[0]

    if x >= x_data[-1]:
        return y_data[-1]

    for i in range(len(x_data) - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            x1, x2 = x_data[i], x_data[i + 1]
            y1, y2 = y_data[i], y_data[i + 1]
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    return y_data[-1]


def integrate_sigma(T0, m, steps=40):
    h = 1.0 / steps
    integral = 0.0

    for i in range(steps + 1):
        z = i * h
        T_z = T0 + (T_w - T0) * (z**m)
        sigma_z = interpolate(T_z, table2_T, table2_sigma)

        f_z = sigma_z * z

        if i == 0 or i == steps:
            weight = 1
        elif i % 2 == 1:
            weight = 4
        else:
            weight = 2

        integral += weight * f_z

    return integral * (h / 3.0)


def get_Rp_and_T0(I):
    abs_I = abs(I)

    T0 = interpolate(abs_I, table1_I, table1_T0)
    m = interpolate(abs_I, table1_I, table1_m)

    integral_val = integrate_sigma(T0, m)

    if integral_val < 1e-12:
        integral_val = 1e-12

    Rp = l_p / (2 * math.pi * (R_radius**2) * integral_val)

    return Rp, T0


def get_derivatives(curr_I, curr_U, mode="standard"):
    if mode == "standard":
        curr_Rp, _ = get_Rp_and_T0(curr_I)
        R_total = R_k + curr_Rp
    elif mode == "zero_R":
        R_total = 0.0
    elif mode == "const_200":
        R_total = 200.0
    else:
        raise ValueError("Неизвестный режим")

    dI = (curr_U - R_total * curr_I) / L_k
    dU = -curr_I / C_k

    return dI, dU


def rk4_step(I, U, h, mode="standard"):
    k1_I, k1_U = get_derivatives(I, U, mode)

    k2_I, k2_U = get_derivatives(
        I + h / 2 * k1_I,
        U + h / 2 * k1_U,
        mode
    )

    k3_I, k3_U = get_derivatives(
        I + h / 2 * k2_I,
        U + h / 2 * k2_U,
        mode
    )

    k4_I, k4_U = get_derivatives(
        I + h * k3_I,
        U + h * k3_U,
        mode
    )

    I_new = I + (h / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
    U_new = U + (h / 6) * (k1_U + 2 * k2_U + 2 * k3_U + k4_U)

    return I_new, U_new


def solve_circuit(t_max, h, mode="standard", I_start=None, U_start=None, t_start=0.0):
    t = t_start
    I = I_0 if I_start is None else I_start
    U = U_0 if U_start is None else U_start

    t_vals, I_vals, U_vals = [t], [I], [U]
    Rp_vals, T0_vals = [], []

    if mode == "standard":
        Rp, T0 = get_Rp_and_T0(I)
    elif mode == "zero_R":
        Rp, T0 = 0.0, 0.0
    elif mode == "const_200":
        Rp, T0 = 200.0, 0.0

    Rp_vals.append(Rp)
    T0_vals.append(T0)

    steps = int(abs((t_max - t_start) / h))

    print(f"Запуск симуляции '{mode}' (шаг: {h:.1e} с, шагов: {steps})...")

    for _ in range(steps):
        I, U = rk4_step(I, U, h, mode)
        t += h

        t_vals.append(t)
        I_vals.append(I)
        U_vals.append(U)

        if mode == "standard":
            Rp, T0 = get_Rp_and_T0(I)
        elif mode == "const_200":
            Rp, T0 = 200.0, 0.0
        else:
            Rp, T0 = 0.0, 0.0

        Rp_vals.append(Rp)
        T0_vals.append(T0)

    return t_vals, I_vals, U_vals, Rp_vals, T0_vals


def run_forward_backward_task():
    h = 5e-6
    t_mid = 605e-6

    # ПРЯМОЙ ХОД: от 0 до 600 мкс
    t_fwd, I_fwd, U_fwd, Rp_fwd, T0_fwd = solve_circuit(
        t_max=t_mid,
        h=h,
        mode="standard",
        I_start=I_0,
        U_start=U_0,
        t_start=0.0
    )

    I_600 = I_fwd[-1]
    U_600 = U_fwd[-1]

    print("\n--- ПРЯМОЙ ХОД ДО 600 мкс ---")
    print(f"t = {t_fwd[-1] * 1e6:.1f} мкс")
    print(f"I(600 мкс) = {I_600:.6f} А")
    print(f"U(600 мкс) = {U_600:.6f} В")

    # ОБРАТНЫЙ ХОД: от 600 мкс до 0 с шагом -h
    t_bwd, I_bwd, U_bwd, Rp_bwd, T0_bwd = solve_circuit(
        t_max=0.0,
        h=-h,
        mode="standard",
        I_start=I_600,
        U_start=U_600,
        t_start=t_mid
    )

    print("\n--- ОБРАТНЫЙ ХОД ОТ 600 мкс ДО 0 ---")
    print(f"t = {t_bwd[-1] * 1e6:.1f} мкс")
    print(f"I_back(0) = {I_bwd[-1]:.6f} А")
    print(f"U_back(0) = {U_bwd[-1]:.6f} В")

    # Для удобного сравнения перевернем обратный ход по времени
    t_bwd_rev = t_bwd[::-1]
    I_bwd_rev = I_bwd[::-1]
    U_bwd_rev = U_bwd[::-1]

    t_fwd_mks = [t * 1e6 for t in t_fwd]
    t_bwd_mks = [t * 1e6 for t in t_bwd]
    t_bwd_rev_mks = [t * 1e6 for t in t_bwd_rev]

    # График тока
    plt.figure(figsize=(10, 5))
    plt.plot(t_fwd_mks, I_fwd, label='Прямой ход I(t)')
    plt.plot(t_bwd_rev_mks, I_bwd_rev, '--', label='Обратный ход I(t)')
    plt.xlabel('Время, мкс')
    plt.ylabel('Ток, А')
    plt.title('Сравнение прямого и обратного хода по току')
    plt.grid()
    plt.legend()
    plt.show()

    # График напряжения
    plt.figure(figsize=(10, 5))
    plt.plot(t_fwd_mks, U_fwd, label='Прямой ход U(t)')
    plt.plot(t_bwd_rev_mks, U_bwd_rev, '--', label='Обратный ход U(t)')
    plt.xlabel('Время, мкс')
    plt.ylabel('Напряжение, В')
    plt.title('Сравнение прямого и обратного хода по напряжению')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_forward_backward_task()