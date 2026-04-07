import math
import matplotlib.pyplot as plt


def f(x, u):
    return x * x + u * u


def picard_3_1(x):
    return x * x * x / 3.0


def picard_3_2(x):
    return picard_3_1(x) + x**7 / 63.0


def picard_3_3(x):
    x3 = x * x * x
    x7 = x**7
    x11 = x**11
    x15 = x**15

    return x3 / 3.0 + x7 / 63.0 + 2.0 * x11 / 2079.0 + x15 / 59535.0


def picard_3_4(x):
    x3 = x * x * x
    x7 = x**7
    x11 = x**11
    x15 = x**15
    x19 = x**19
    x23 = x**23
    x27 = x**27
    x31 = x**31

    return (
        x3 / 3.0
        + x7 / 63.0
        + 2.0 * x11 / 2079.0
        + x15 / 59535.0
        + 2.0 * x19 / 93555.0
        + 2.0 * x23 / 3393495.0
        + 2.0 * x27 / 3341878155.0
        + x31 / 109876902975.0
    )


def euler(x0, u0, h, n):
    x = [0.0] * n
    u = [0.0] * n

    x[0] = x0
    u[0] = u0

    for i in range(n - 1):
        u[i + 1] = u[i] + h * f(x[i], u[i])
        x[i + 1] = x[i] + h

    return x, u


def find_xmax(h_base):
    target_rel_error = 1e-4
    print(
        f"Базовый шаг h = {h_base:.6e}, "
        f"целевая относительная точность = {target_rel_error:.0e}"
    )

    x_check = 0.0
    step_check = 0.02
    x_last_good = 0.0

    while x_check < 2.5:
        x_end = x_check + step_check

        n_fine = int(x_end / h_base) + 1
        _, u = euler(0, 0, h_base, n_fine)
        u_final = u[-1]

        n_coarse = int(x_end / (2 * h_base)) + 1
        _, u_vals_coarse = euler(0, 0, 2 * h_base, n_coarse)
        u_coarse = u_vals_coarse[-1]

        if abs(u_final) > 1e-8:
            rel_error = abs(u_final - u_coarse) / abs(u_final)
        else:
            rel_error = abs(u_final - u_coarse)

        if rel_error > target_rel_error and x_check > 0.1:
            print(f"\nОтносительная погрешность Эйлера превышена при x = {x_end:.4f}")
            print(f"  relError = {rel_error:.2e} > {target_rel_error:.0e}")
            return x_end

        x_last_good = x_end
        print(x_check, x_end)
        x_check = x_end

    return x_last_good


def task_03():
    print("Уравнение: u' = x2 + u2, u(0) = 0")

    h_euler = 5e-7
    h_output = 0.01

    # xmax = find_xmax(h_euler)
    # print(xmax)
    xmax = 2.0

    n = int(xmax / h_euler) + 1
    x_vals, u_vals = euler(0, 0, h_euler, n)

    print("\n========== РЕЗУЛЬТАТЫ ==========")
    print(f"Интервал [0, {xmax:f}] с шагом вывода {h_output:f}")
    print("=" * 93)
    print(
        f"{'x':<8} | {'Пикар P1':<8} | {'Пикар P2':<8} | "
        f"{'Пикар P3':<8} | {'Пикар P4':<8} | {'Эйлер':<8}"
    )
    print("-" * 93)

    x = 0.0
    while x <= xmax + 1e-8:
        idx = int(x / h_euler + 0.5)
        if idx >= len(x_vals):
            idx = len(x_vals) - 1

        p1 = picard_3_1(x)
        p2 = picard_3_2(x)
        p3 = picard_3_3(x)
        p4 = picard_3_4(x)
        u_e = u_vals[idx]

        print(
            f"{x:<8.2f} | {p1:<8.4f} | {p2:<8.4f} | "
            f"{p3:<8.4f} | {p4:<8.4f} | {u_e:<8.4f}"
        )
        x += h_output

    if abs(xmax % h_output) > 1e-8:
        x = xmax
        idx = len(x_vals) - 1
        p1 = picard_3_1(x)
        p2 = picard_3_2(x)
        p3 = picard_3_3(x)
        p4 = picard_3_4(x)
        u_e = u_vals[idx]

        print(
            f"{x:<8.2f} | {p1:<12.8f} | {p2:<12.8f} | "
            f"{p3:<12.8f} | {p4:<12.8f} | {u_e:<12.8f}"
        )

    print("=" * 93)

    x_plot = []
    p1_plot = []
    p2_plot = []
    p3_plot = []
    p4_plot = []
    euler_plot = []

    x = 0.0
    while x <= xmax + 1e-8:
        idx = int(x / h_euler + 0.5)
        if idx >= len(x_vals):
            idx = len(x_vals) - 1

        x_plot.append(x)
        p1_plot.append(picard_3_1(x))
        p2_plot.append(picard_3_2(x))
        p3_plot.append(picard_3_3(x))
        p4_plot.append(picard_3_4(x))
        euler_plot.append(u_vals[idx])
        x += h_output

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, euler_plot, label='Эйлер', linewidth=2)
    plt.plot(x_plot, p1_plot, label='Пикар P1', linestyle='--')
    plt.plot(x_plot, p2_plot, label='Пикар P2', linestyle='--')
    plt.plot(x_plot, p3_plot, label='Пикар P3', linestyle='--')
    plt.plot(x_plot, p4_plot, label='Пикар P4', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.axvline(0, color='black', lw=1)
    plt.axhline(0, color='black', lw=0.5)
    plt.title("Сравнение методов решения ОДУ: u' = x² + u²")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('result3.png', dpi=300, bbox_inches='tight')
    print("\nГрафик сохранён в файл result3.png")


if __name__ == "__main__":
    task_03()