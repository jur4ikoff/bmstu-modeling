import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 1
# ─────────────────────────────────────────────────────────────


def task1_galerkin(N=3):
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(40)
    xq = 0.5 * (nodes + 1)
    wq = 0.5 * weights

    def phi(n, x):
        return x**n * (1.0 - x)

    def Lphi(n, x):
        ddp = n * (n - 1) * x ** (n - 2) * (1.0 - x) - 2.0 * n * x ** (n - 1)
        dp = n * x ** (n - 1) * (1.0 - x) - x**n
        pp = x**n * (1.0 - x)
        return ddp - 2.0 * x * dp + 2.0 * pp

    A = np.zeros((N, N))
    b = np.zeros(N)
    for i in range(N):
        ni = i + 1
        phi_i = phi(ni, xq)
        for j in range(N):
            A[i, j] = np.dot(wq, Lphi(j + 1, xq) * phi_i)
        b[i] = np.dot(wq, xq * phi_i)

    c = np.linalg.solve(A, b)
    x = np.linspace(0.0, 1.0, 300)
    u = sum(c[i] * x ** (i + 1) * (1.0 - x) for i in range(N))
    return x, u


def task1_fdm(N=200):
    x = np.linspace(0.0, 1.0, N + 1)
    h = x[1] - x[0]

    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    d = np.zeros(N + 1)

    for i in range(1, N):
        xi = x[i]
        a[i] = 1.0 / h**2 + xi / h
        b[i] = -2.0 / h**2 + 2.0
        c[i] = 1.0 / h**2 - xi / h
        d[i] = xi

    b[0] = -1.0
    c[0] = 1.0
    d[0] = 0.0
    a[N] = 1.0 / h
    b[N] = -1.0 / h
    d[N] = 1.0

    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]
    for i in range(1, N + 1):
        denom = b[i] + a[i] * alpha[i - 1]
        alpha[i] = -c[i] / denom if i < N else 0.0
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

    u = np.zeros(N + 1)
    u[N] = beta[N]
    for i in range(N - 1, -1, -1):
        u[i] = alpha[i] * u[i + 1] + beta[i]
    return x, u


def plot_task1():
    xg, ug = task1_galerkin()
    xn, un = task1_fdm()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xg, ug, "b-", lw=2, label="Метод Галёркина (N=3)")
    ax.plot(xn, un, "r--", lw=2, label="Метод прогонки (МКР, N=200)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Задача 1: u'' - 2xu' + 2u = x,  u(0)=0, u'(1)=1")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("task1.png", dpi=150)
    plt.close()
    print("Задача 1 — OK")


# ─────────────────────────────────────────────────────────────
#  Общие параметры задач 2 и 3
# ─────────────────────────────────────────────────────────────

_T_data = np.array([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], dtype=float)
_k_v1 = np.array(
    [
        8.200e-3,
        2.768e-2,
        6.560e-2,
        1.281e-1,
        2.214e-1,
        3.516e-1,
        5.248e-1,
        7.472e-1,
        1.025e0,
    ]
)
_k_v2 = np.array(
    [1.600e0, 5.400e0, 1.280e1, 2.500e1, 4.320e1, 6.860e1, 1.024e2, 1.458e2, 2.000e2]
)
_lnT = np.log(_T_data)
_p1fit = np.polyfit(_lnT, np.log(_k_v1), 1)
_p2fit = np.polyfit(_lnT, np.log(_k_v2), 1)

R = 0.35
Tw = 2000.0
T0 = 1e4
p_exp = 4
c_light = 3e10


def k_func(T, variant):
    lT = np.log(np.clip(np.asarray(T, dtype=float), 1e-10, None))
    return np.exp(np.polyval(_p1fit if variant == 1 else _p2fit, lT))


def T_field(r):
    return (Tw - T0) * (np.asarray(r, dtype=float) / R) ** p_exp + T0


def u_planck(r):
    T = T_field(r)
    return 3.084e-4 / (np.exp(4.799e4 / T) - 1.0)


# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 2  —  метод стрельбы
# ─────────────────────────────────────────────────────────────
#
#  Замена переменных: F = -1/(3k) * du/dr
#  Система 1-го порядка:
#    du/dr = -3k(r)*F
#    dF/dr = -k(r)*(u - u_p(r)) - F/r
#
#  ГУ: r=0 -> F=0,  r=R -> F(R) = 0.39*u(R)
#  Стрельба: u(0) = chi * u_p(0)


def _ode(r, y, variant):
    u, F = y
    r = max(r, 1e-12)
    k = float(k_func(T_field(r), variant))
    return [-3.0 * k * F, -k * (u - float(u_planck(r))) - F / r]


# ─────────────────────────────────────────────────────────────
#  Собственные реализации РК2 и РК4
#
#  Обе функции интегрируют систему du/dr = f(r, y)
#  от r=r0 до r=R с шагом h и возвращают
#  массивы r_arr, u_arr, F_arr.
# ─────────────────────────────────────────────────────────────


def _rk2(chi, variant, N_steps=1000):
    """
    Метод Рунге-Кутта 2-го порядка (метод средней точки).

    Формула:
        k1 = f(rₙ, yₙ)
        k2 = f(rₙ + h/2, yₙ + h/2 * k1)
        yₙ₊₁ = yₙ + h * k2

    Погрешность на шаге: O(h³)
    Погрешность на всём отрезке: O(h²)
    """
    h = (R - 1e-10) / N_steps
    r = 1e-10
    y = np.array([chi * float(u_planck(1e-10)), 0.0])

    r_arr = [r]
    u_arr = [y[0]]
    F_arr = [y[1]]

    for _ in range(N_steps):
        f = _ode(r, y, variant)
        k1 = np.array(f)

        r_mid = r + 0.5 * h
        y_mid = y + 0.5 * h * k1
        k2 = np.array(_ode(r_mid, y_mid, variant))

        y = y + h * k2
        r = r + h

        r_arr.append(r)
        u_arr.append(y[0])
        F_arr.append(y[1])

    return np.array(r_arr), np.array(u_arr), np.array(F_arr)


def _rk4(chi, variant, N_steps=1000):
    """
    Метод Рунге-Кутта 4-го порядка (классический).

    Формула:
        k1 = f(rₙ,       yₙ)
        k2 = f(rₙ + h/2, yₙ + h/2 * k1)
        k3 = f(rₙ + h/2, yₙ + h/2 * k2)
        k4 = f(rₙ + h,   yₙ + h   * k3)
        yₙ₊₁ = yₙ + h/6 * (k1 + 2k2 + 2k3 + k4)

    Погрешность на шаге: O(h⁵)
    Погрешность на всём отрезке: O(h⁴)
    """
    h = (R - 1e-10) / N_steps
    r = 1e-10
    y = np.array([chi * float(u_planck(1e-10)), 0.0])

    r_arr = [r]
    u_arr = [y[0]]
    F_arr = [y[1]]

    for _ in range(N_steps):
        k1 = np.array(_ode(r, y, variant))
        k2 = np.array(_ode(r + 0.5 * h, y + 0.5 * h * k1, variant))
        k3 = np.array(_ode(r + 0.5 * h, y + 0.5 * h * k2, variant))
        k4 = np.array(_ode(r + h, y + h * k3, variant))

        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        r = r + h

        r_arr.append(r)
        u_arr.append(y[0])
        F_arr.append(y[1])

    return np.array(r_arr), np.array(u_arr), np.array(F_arr)


def _bc_res_rk(chi, variant, rk_func, N_steps):
    r_arr, u_arr, F_arr = rk_func(chi, variant, N_steps)
    return F_arr[-1] - 0.39 * u_arr[-1]


def _bc_res_rk2(chi, variant, N_steps=1000):
    return _bc_res_rk(chi, variant, _rk2, N_steps)


def _bc_res_rk4(chi, variant, N_steps=1000):
    return _bc_res_rk(chi, variant, _rk4, N_steps)


# ─────────────────────────────────────────────────────────────


def _shoot(chi, variant, method="RK45", max_step_div=500):
    u0 = chi * float(u_planck(1e-10))
    return solve_ivp(
        _ode,
        [1e-10, R],
        [u0, 0.0],
        args=(variant,),
        method=method,
        dense_output=True,
        max_step=R / max_step_div,
        rtol=1e-8,
        atol=1e-10,
    )


def _bc_res(chi, variant, method="RK45", ms=200):
    sol = _shoot(chi, variant, method=method, max_step_div=ms)
    uR, FR = sol.y[0, -1], sol.y[1, -1]
    return FR - 0.39 * uR


def solve_shooting(variant):
    if variant == 1:
        chis = np.linspace(0.05, 0.5, 30)
        vals = [_bc_res(c, variant) for c in chis]
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0:
                chi_star = brentq(
                    _bc_res, chis[i], chis[i + 1], args=(variant,), xtol=1e-10
                )
                return _shoot(chi_star, variant, max_step_div=1000), chi_star
        raise RuntimeError("Корень не найден для варианта 1")
    else:
        chi_star = brentq(
            _bc_res, 0.9999985, 0.9999995, args=(variant, "Radau", 50), xtol=1e-12
        )
        return _shoot(chi_star, variant, method="Radau", max_step_div=200), chi_star


def solve_shooting_rk2(variant, N_steps=1000):
    """Метод стрельбы с РК2 (только вариант 1 — нежёсткий)."""
    chis = np.linspace(0.05, 0.5, 30)
    vals = [_bc_res_rk2(c, variant, N_steps) for c in chis]
    for i in range(len(vals) - 1):
        if vals[i] * vals[i + 1] < 0:
            chi_star = brentq(
                _bc_res_rk2, chis[i], chis[i + 1], args=(variant, N_steps), xtol=1e-8
            )
            return _rk2(chi_star, variant, N_steps), chi_star
    raise RuntimeError("РК2: корень не найден")


def solve_shooting_rk4(variant, N_steps=1000):
    """Метод стрельбы с РК4 (только вариант 1 — нежёсткий)."""
    chis = np.linspace(0.05, 0.5, 30)
    vals = [_bc_res_rk4(c, variant, N_steps) for c in chis]
    for i in range(len(vals) - 1):
        if vals[i] * vals[i + 1] < 0:
            chi_star = brentq(
                _bc_res_rk4, chis[i], chis[i + 1], args=(variant, N_steps), xtol=1e-8
            )
            return _rk4(chi_star, variant, N_steps), chi_star
    raise RuntimeError("РК4: корень не найден")


def plot_task2():
    # ── График 1: основное решение (solve_ivp) для двух вариантов ──
    for variant in (1, 2):
        sol, chi = solve_shooting(variant)
        r_arr = sol.t
        u_arr = sol.y[0]
        F_arr = sol.y[1]
        up_arr = u_planck(r_arr)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(r_arr, u_arr, lw=2, label="u(r)")
        ax.plot(r_arr, up_arr, lw=2, ls="--", label="u_p(r)")
        ax.plot(r_arr, F_arr, lw=2, ls="-.", label="F(r)")
        ax.set_title(f"Задача 2, Вариант {variant}  (chi={chi:.10f})")
        ax.set_xlabel("r, см")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"task2_variant_{variant}.png", dpi=150)
        plt.close()
        print(f"Задача 2, Вариант {variant}: chi* = {chi:.12f}")

    # ── График 2: сравнение РК2 / РК4 / solve_ivp (только вариант 1) ──
    # Для варианта 2 обычные явные РК2/РК4 плохо подходят: система жёсткая,
    # поэтому выше для него используется solve_ivp с методом Radau.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    variant = 1
    sol_ref, chi_ref = solve_shooting(variant)

    (r2, u2, F2), chi2 = solve_shooting_rk2(variant, N_steps=1000)
    (r4, u4, F4), chi4 = solve_shooting_rk4(variant, N_steps=1000)

    print(f"\nСравнение методов (Вариант 1):")
    print(f"  solve_ivp (RK45): chi*={chi_ref:.8f}")
    print(f"  РК2 N=1000:       chi*={chi2:.8f}")
    print(f"  РК4 N=1000:       chi*={chi4:.8f}")

    ax = axes[0]
    ax.plot(sol_ref.t, sol_ref.y[0], "k-", lw=2.5, label="u(r) solve_ivp")
    ax.plot(r2, u2, "b--", lw=2, label="u(r) РК2")
    ax.plot(r4, u4, "r-.", lw=2, label="u(r) РК4")
    ax.set_title("Задача 2, Вариант 1: сравнение РК2 / РК4 / solve_ivp")
    ax.set_xlabel("r, см")
    ax.legend()
    ax.grid(True)

    ax2 = axes[1]
    ax2.plot(sol_ref.t, sol_ref.y[1], "k-", lw=2.5, label="F(r) solve_ivp")
    ax2.plot(r2, F2, "b--", lw=2, label="F(r) РК2")
    ax2.plot(r4, F4, "r-.", lw=2, label="F(r) РК4")
    ax2.set_title("Задача 2, Вариант 1: поток F(r)")
    ax2.set_xlabel("r, см")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("task2_rk_compare.png", dpi=150)
    plt.close()

    # ── График 3: сходимость РК2 и РК4 по числу шагов ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    N_list = [50, 100, 200, 500, 1000, 2000]
    err2, err4 = [], []

    for Ns in N_list:
        (r2s, u2s, _), _ = solve_shooting_rk2(variant, Ns)
        (r4s, u4s, _), _ = solve_shooting_rk4(variant, Ns)
        u2_at_ref = np.interp(sol_ref.t, r2s, u2s)
        u4_at_ref = np.interp(sol_ref.t, r4s, u4s)
        err2.append(np.max(np.abs(u2_at_ref - sol_ref.y[0])))
        err4.append(np.max(np.abs(u4_at_ref - sol_ref.y[0])))

    h_list = [R / Ns for Ns in N_list]

    ax = axes[0]
    ax.loglog(h_list, err2, "bo-", lw=2, ms=7, label="РК2: O(h²)")
    ax.loglog(h_list, err4, "rs-", lw=2, ms=7, label="РК4: O(h⁴)")
    ref2 = err2[0] * (np.array(h_list) / h_list[0]) ** 2
    ref4 = err4[0] * (np.array(h_list) / h_list[0]) ** 4
    ax.loglog(h_list, ref2, "b--", lw=1, alpha=0.5, label="~h²")
    ax.loglog(h_list, ref4, "r--", lw=1, alpha=0.5, label="~h⁴")
    ax.set_xlabel("Шаг h")
    ax.set_ylabel("Максимальная погрешность")
    ax.set_title("Сходимость РК2 и РК4 (Вариант 1)")
    ax.legend()
    ax.grid(True)

    ax2 = axes[1]
    ax2.loglog(N_list, err2, "bo-", lw=2, ms=7, label="РК2")
    ax2.loglog(N_list, err4, "rs-", lw=2, ms=7, label="РК4")
    ax2.set_xlabel("Число шагов N")
    ax2.set_ylabel("Максимальная погрешность")
    ax2.set_title("Погрешность vs число шагов")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("task2_convergence.png", dpi=150)
    plt.close()

    print(
        "\nЗадача 2 — OK. Графики: task2_variant_1.png, task2_variant_2.png, task2_rk_compare.png, task2_convergence.png"
    )


# ─────────────────────────────────────────────────────────────
#  ЗАДАЧА 3  —  метод конечных разностей
# ─────────────────────────────────────────────────────────────
#
#  ИИМ-дискретизация: 1/r * d/dr[r/(3k)*du/dr] = 3k(u - u_p)
#  r=0 : u_1 - u_0 = 0  (du/dr = 0, симметрия)
#  r=R : -1/(3k)*du/dr = 0.39*u(R)
#         du/dr аппроксимирована формулой 2-го порядка


def solve_fdm(variant, N=300):
    r = np.linspace(0.0, R, N + 1)
    h = r[1] - r[0]
    k_r = k_func(T_field(r), variant)
    up_r = u_planck(r)

    a_c = np.zeros(N + 1)
    b_c = np.zeros(N + 1)
    c_c = np.zeros(N + 1)
    d_c = np.zeros(N + 1)

    for i in range(1, N):
        ri = r[i]
        kp = 0.5 * (k_r[i] + k_r[i + 1])
        km = 0.5 * (k_r[i] + k_r[i - 1])
        rp = ri + 0.5 * h
        rm = ri - 0.5 * h
        a_c[i] = rm / (3.0 * km * ri * h**2)
        c_c[i] = rp / (3.0 * kp * ri * h**2)
        b_c[i] = -(a_c[i] + c_c[i]) - 3.0 * k_r[i]
        d_c[i] = -3.0 * k_r[i] * up_r[i]

    b_c[0] = 1.0
    c_c[0] = -1.0
    d_c[0] = 0.0

    km_N = 0.5 * (k_r[N] + k_r[N - 1])
    rm_N = R - 0.5 * h
    coeff2 = 3.0 / (2.0 * h * 3.0 * k_r[N])
    a_c[N] = rm_N / (3.0 * km_N * R * h**2)
    b_c[N] = -(a_c[N]) - 3.0 * k_r[N] - 0.39 * coeff2
    d_c[N] = -3.0 * k_r[N] * up_r[N]

    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)
    alpha[0] = -c_c[0] / b_c[0]
    beta[0] = d_c[0] / b_c[0]
    for i in range(1, N + 1):
        denom = b_c[i] + a_c[i] * alpha[i - 1]
        alpha[i] = -c_c[i] / denom if i < N else 0.0
        beta[i] = (d_c[i] - a_c[i] * beta[i - 1]) / denom

    u = np.zeros(N + 1)
    u[N] = beta[N]
    for i in range(N - 1, -1, -1):
        u[i] = alpha[i] * u[i + 1] + beta[i]

    F = np.zeros(N + 1)
    F[0] = 0.0
    for i in range(1, N):
        F[i] = -1.0 / (3.0 * k_r[i]) * (u[i + 1] - u[i - 1]) / (2.0 * h)
    F[N] = -1.0 / (3.0 * k_r[N]) * (3.0 * u[N] - 4.0 * u[N - 1] + u[N - 2]) / (2.0 * h)

    divF = np.zeros(N + 1)
    for i in range(1, N):
        divF[i] = (r[i] * F[i] - r[i - 1] * F[i - 1]) / (h * r[i])
    divF[0] = 2.0 * F[1] / h
    divF[N] = (r[N] * F[N] - r[N - 1] * F[N - 1]) / (h * R)

    integrand = c_light / R * k_r * (up_r - u) * r
    cumint = np.concatenate(
        [[0.0], np.cumsum(0.5 * h * (integrand[:-1] + integrand[1:]))]
    )
    F_int = np.zeros(N + 1)
    F_int[1:] = cumint[1:] / r[1:]

    F_R = {
        "Правая 1-го порядка": -(u[N] - u[N - 1]) / (h * 3.0 * k_r[N]),
        "2-й порядок точности": F[N],
        "Интегрирование": F_int[N],
        "Из краевого условия": 0.39 * u[N],
    }
    return r, u, up_r, F, divF, F_R


def plot_task3():
    variant = 1
    r, u, up, F, divF, F_R = solve_fdm(variant)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(r, u, lw=2, label="u(r)")
    ax.plot(r, up, lw=2, ls="--", label="u_p(r)")
    ax.set_title("Задача 3, Вариант 1: u(r) и u_p(r)")
    ax.set_xlabel("r, см")
    ax.legend()
    ax.grid(True)

    ax2 = axes[1]
    ax2.plot(r, F, lw=2, label="F(r)")
    ax2.plot(r, divF, lw=2, ls="--", label="divF(r)")
    ax2.set_title("Задача 3, Вариант 1: F(r) и divF(r)")
    ax2.set_xlabel("r, см")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("task3.png", dpi=150)
    plt.close()

    print(f"\nЗадача 3, Вариант 1 — F(R) четырьмя способами:")
    for name, val in F_R.items():
        print(f"  {name:30s}: {val:.6e}")
    print("\nЗадача 3 — OK")


# ─────────────────────────────────────────────────────────────
#  Сравнение задач 2 и 3
# ─────────────────────────────────────────────────────────────


def compare_tasks23():
    variant = 1
    sol, chi = solve_shooting(variant)
    r_s, u_s = sol.t, sol.y[0]
    r_f, u_f, _, _, _, _ = solve_fdm(variant)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_s, u_s, "b-", lw=2, label="u(r) стрельба")
    ax.plot(r_f, u_f, "r--", lw=2, label="u(r) МКР")
    ax.set_title("Сравнение задач 2 и 3, Вариант 1")
    ax.set_xlabel("r, см")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("task23_compare.png", dpi=150)
    plt.close()

    delta = abs(u_s[-1] - u_f[-1]) / max(abs(u_s[-1]), 1e-30) * 100
    print(
        f"Вариант 1: u(R) стрельба={u_s[-1]:.4e}, "
        f"МКР={u_f[-1]:.4e}, delta={delta:.2f}%"
    )
    print("Сравнение — OK")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Задача 1 ===")
    plot_task1()

    print("\n=== Задача 2 ===")
    plot_task2()

    print("\n=== Задача 3 ===")
    plot_task3()

    print("\n=== Сравнение задач 2 и 3 ===")
    compare_tasks23()

    print("\nГотово. Графики сохранены в текущей папке.")
