import sys
import math
import os

import plotly.graph_objects as go


# ==================== Задача 1 ====================


def taylor_4(x: float) -> float:
    """Приближение рядом Тейлора (4 члена): u(x) approx 1 - x/2 + x^2/24 - x^3/720
    """
    return 1.0 - x / 2.0 + x * x / 24.0 - x * x * x / 720.0


def exact_task1(x: float) -> float:
    """Точное аналитическое решение: u(x) = cos(sqrt(x))"""
    return math.cos(math.sqrt(x))


def euler_task1(x_end: float, step: float) -> list[tuple[float, float]]:
    """Метод Эйлера для системы:
      y'_1 = y_2
      y'_2 = -(2 y_2 + y_1) / (4x)

    Особенность при x=0: стартуем с x_0 = epsilon, используя ряд для начальных значений.
    """
    u = taylor_4(step)

    u_deriv = -0.5 + step / 12.0 - step * step / 240.0

    results = [(0.0, 1.0), (step, u)]
    x = step

    while x + step <= x_end + 1e-12:
        du = u_deriv
        du_deriv = -(2.0 * u_deriv + u) / (4.0 * x)

        u += step * du
        u_deriv += step * du_deriv
        x += step

        results.append((x, u))

    return results


def run_task1():
    print(">>> Задача 1: 4xu'' + 2u' + u = 0\n")

    step = 0.01
    x_end = 10.0

    n_points = int(x_end / step) + 1
    xs = [i * step for i in range(n_points)]

    euler_data = euler_task1(x_end, step)

    print(f"{'x':>8} | {'Ряд Тейлора':>14} | {'Эйлер':>14} | {'Точное':>14}")
    print("-" * 60)
    for x_value in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        taylor_value = taylor_4(x_value)
        exact_value = exact_task1(x_value)

        idx = round(x_value / step)
        euler_value = euler_data[idx][1] if idx < len(euler_data) else float("nan")

        print(
            f"{x_value:>8.2f} | {taylor_value:>14.2f} | {euler_value:>14.2f} | {exact_value:>14.2f}"
        )

    # График
    ys_exact = [exact_task1(x) for x in xs]
    ys_taylor = [taylor_4(x) for x in xs]
    xs_euler = [pt[0] for pt in euler_data]
    ys_euler = [pt[1] for pt in euler_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys_exact, name="cos(sqrt(x)) -- точное"))
    fig.add_trace(go.Scatter(x=xs, y=ys_taylor, name="Ряд Тейлора (4 члена)"))
    fig.add_trace(go.Scatter(x=xs_euler, y=ys_euler, name="Метод Эйлера"))
    fig.update_layout(title="Задача 1: 4xu'' + 2u' + u = 0")

    os.makedirs("plots", exist_ok=True)
    fig.write_html("plots/task1.html")
    print("\nГрафик сохранён в plots/task1.html")


# ==================== Задача 2 ====================


def exact_task2(u: float) -> float:
    """Точное решение (обратная функция): x(u) = e^(u^2) - (u^2 + 1) / 2"""
    return math.exp(u * u) - (u * u + 1.0) / 2.0


def picard_1(u: float) -> float:
    """Приближение Пикара #1: x(u) = 0.5 + u^2/2 + u^4/4"""
    u2 = u * u
    return 0.5 + u2 / 2.0 + u2 * u2 / 4.0


def picard_2(u: float) -> float:
    """Приближение Пикара #2: x(u) = 0.5 + u^2/2 + u^4/2 + u^6/12"""
    u2 = u * u
    return 0.5 + u2 / 2.0 + u2 * u2 / 2.0 + u2 * u2 * u2 / 12.0


def picard_3(u: float) -> float:
    """Приближение Пикара #3: x(u) = 0.5 + u^2/2 + u^4/2 + u^6/6 + u^8/48"""
    u2 = u * u
    u4 = u2 * u2
    return 0.5 + u2 / 2.0 + u4 / 2.0 + u4 * u2 / 6.0 + u4 * u4 / 48.0


def picard_4(u: float) -> float:
    """Приближение Пикара #4: x(u) = 0.5 + u^2/2 + u^4/2 + u^6/6 + u^8/24 + u^10/240"""
    u2 = u * u
    u4 = u2 * u2
    return (
        0.5
        + u2 / 2.0
        + u4 / 2.0
        + u4 * u2 / 6.0
        + u4 * u4 / 24.0
        + u4 * u4 * u2 / 240.0
    )


def run_task2():
    print("\n>>> Задача 2: 1 - 2xu u' = u^3 u'\n")

    u_values = [i * 0.1 for i in range(21)]

    print(
        f"{'u':>6} | {'Пикар 1':>12} | {'Пикар 2':>12} | {'Пикар 3':>12} | {'Пикар 4':>12} | {'Точное':>12}"
    )
    print("-" * 81)
    for u_value in u_values:
        p1 = picard_1(u_value)
        p2 = picard_2(u_value)
        p3 = picard_3(u_value)
        p4 = picard_4(u_value)
        exact_value = exact_task2(u_value)

        print(
            f"{u_value:>6.2f} | {p1:>12.2f} | {p2:>12.2f} | {p3:>12.2f} | {p4:>12.2f} | {exact_value:>12.2f}"
        )

    # График
    us = [i * 0.01 for i in range(201)]
    xs_exact = [exact_task2(u) for u in us]
    xs_p1 = [picard_1(u) for u in us]
    xs_p2 = [picard_2(u) for u in us]
    xs_p3 = [picard_3(u) for u in us]
    xs_p4 = [picard_4(u) for u in us]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=us, y=xs_exact, name="Точное"))
    fig.add_trace(go.Scatter(x=us, y=xs_p1, name="Пикар 1"))
    fig.add_trace(go.Scatter(x=us, y=xs_p2, name="Пикар 2"))
    fig.add_trace(go.Scatter(x=us, y=xs_p3, name="Пикар 3"))
    fig.add_trace(go.Scatter(x=us, y=xs_p4, name="Пикар 4"))
    fig.update_layout(
        title="Задача 2: x(u), метод Пикара",
        xaxis_title="u",
        yaxis_title="x",
    )

    fig.write_html("plots/task2.html")
    print("\nГрафик сохранён в plots/task2.html")


# ==================== Задача 3 ====================


def picard_task3_1(x: float) -> float:
    """Приближение Пикара #1 для u' = x^2 + u^2: y^(1)(x) = x^3/3"""
    return x**3 / 3.0


def picard_task3_2(x: float) -> float:
    """Приближение Пикара #2: y^(2)(x) = x^3/3 + x^7/63"""
    return x**3 / 3.0 + x**7 / 63.0


def picard_task3_3(x: float) -> float:
    """Приближение Пикара #3: y^(3)(x) = x^3/3 + x^7/63 + 2x^11/(189*11) + x^15/(63^2*15)"""
    return (
        x**3 / 3.0
        + x**7 / 63.0
        + 2.0 * x**11 / (189.0 * 11.0)
        + x**15 / (63.0 * 63.0 * 15.0)
    )


def picard_task3_4(x: float) -> float:
    """Приближение Пикара #4 (члены до x^23)"""
    # Знаменатели членов y^(3): 2x^11 / denom_x11, x^15 / denom_x15
    denom_x11 = 189.0 * 11.0
    denom_x15 = 63.0 * 63.0 * 15.0

    # Коэффициенты [y^(3)]^2 при t^14, t^18, t^22, делённые на (степень + 1)
    coeff_x15 = (1.0 / (63.0 * 63.0) + 4.0 / (3.0 * denom_x11)) / 15.0
    coeff_x19 = (2.0 / (3.0 * denom_x15) + 4.0 / (63.0 * denom_x11)) / 19.0
    coeff_x23 = (4.0 / (denom_x11 * denom_x11) + 2.0 / (63.0 * denom_x15)) / 23.0

    return (
        x**3 / 3.0
        + x**7 / 63.0
        + 2.0 * x**11 / denom_x11
        + coeff_x15 * x**15
        + coeff_x19 * x**19
        + coeff_x23 * x**23
    )


def euler_task3(x_end: float, step: float) -> list[tuple[float, float]]:
    """Метод Эйлера для u' = x^2 + u^2, u(0) = 0"""
    u = 0.0
    x = 0.0
    results = [(x, u)]

    while x + step <= x_end + 1e-12:
        deriv = x * x + u * u
        u += step * deriv
        x += step

        if not math.isfinite(u):
            break
        results.append((x, u))

    return results


def find_x_max(step: float, eps: float) -> float:
    """Поиск x_max по правилу Рунге: сравнение Эйлера с шагом step и step/2.
    Относительная погрешность < eps.
    """
    x_limit = 3.0
    data_h = euler_task3(x_limit, step)
    data_h2 = euler_task3(x_limit, step / 2.0)

    x_max = 0.0

    for i, (x, y_h) in enumerate(data_h):
        idx_h2 = i * 2
        if idx_h2 >= len(data_h2):
            break
        _, y_h2 = data_h2[idx_h2]

        if not math.isfinite(y_h) or not math.isfinite(y_h2):
            break

        # Используем max(|y_{h/2}|, abs_tol) в знаменателе,
        # чтобы избежать деления на ~0 при малых значениях решения
        abs_tol = 1e-6
        rel_error = abs(y_h - y_h2) / max(abs(y_h2), abs_tol)
        if rel_error < eps:
            x_max = x

    return x_max


def run_task3():
    print("\n>>> Задача 3: u' = x² + u², u(0) = 0\n")

    step = 1e-4
    x_max = find_x_max(step, 1e-4)
    print(f"x_max (правило Рунге, eps = 1e-4, h = {step}): {x_max:.4f}")

    euler_data = euler_task3(x_max, step / 2.0)

    # Таблица
    display_step = 0.1
    n_display = int(x_max / display_step)

    print(
        f"\n{'x':>6} | {'Пикар 1':>14} | {'Пикар 2':>14} | {'Пикар 3':>14} | {'Пикар 4':>14} | {'Эйлер':>14}"
    )
    print("-" * 91)

    for i in range(n_display + 1):
        x_val = i * display_step

        p1 = picard_task3_1(x_val)
        p2 = picard_task3_2(x_val)
        p3 = picard_task3_3(x_val)
        p4 = picard_task3_4(x_val)

        euler_step = step / 2.0
        idx = round(x_val / euler_step)
        euler_val = euler_data[idx][1] if idx < len(euler_data) else float("nan")

        print(
            f"{x_val:>6.2f} | {p1:>14.3f} | {p2:>14.3f} | {p3:>14.3f} | {p4:>14.3f} | {euler_val:>14.3f}"
        )

    # График
    plot_step = 0.001
    n_plot = int(x_max / plot_step)
    xs_plot = [i * plot_step for i in range(n_plot + 1)]

    ys_p1 = [picard_task3_1(x) for x in xs_plot]
    ys_p2 = [picard_task3_2(x) for x in xs_plot]
    ys_p3 = [picard_task3_3(x) for x in xs_plot]
    ys_p4 = [picard_task3_4(x) for x in xs_plot]

    xs_euler = [pt[0] for pt in euler_data]
    ys_euler = [pt[1] for pt in euler_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_plot, y=ys_p1, name="Пикар 1"))
    fig.add_trace(go.Scatter(x=xs_plot, y=ys_p2, name="Пикар 2"))
    fig.add_trace(go.Scatter(x=xs_plot, y=ys_p3, name="Пикар 3"))
    fig.add_trace(go.Scatter(x=xs_plot, y=ys_p4, name="Пикар 4"))
    fig.add_trace(go.Scatter(x=xs_euler, y=ys_euler, name="Эйлер"))
    fig.update_layout(title="Задача 3: u' = x² + u², u(0) = 0")

    fig.write_html("plots/task3.html")
    print("\nГрафик сохранён в plots/task3.html")


def main():
    run_task1()
    run_task2()
    run_task3()

if __name__ == "__main__":
    main()
