import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_trapz = getattr(np, 'trapezoid', None) or np.trapz

C_LIGHT = 3.0e10  # см/с

T_TABLE = np.array([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000], dtype=float)
SIGMA_TABLE = np.array([0.309e-3, 0.309e-2, 0.309e-1, 0.270, 2.05, 6.06, 12.0, 19.9, 29.6, 41.1, 54.1])
LAMBDA_TABLE = np.array([0.381e-3, 0.381e-3, 0.381e-3, 0.448e-3, 0.577e-3, 0.733e-3, 1.31e-3, 2.18e-3, 3.58e-3, 5.62e-3, 8.32e-3])
CV_TABLE = np.array([1.90e-3, 1.90e-3, 0.95e-3, 0.75e-3, 0.64e-3, 0.61e-3, 0.66e-3, 0.66e-3, 1.15e-3, 1.79e-3, 2.02e-3])

T_K_TABLE = np.array([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], dtype=float)
K_TABLE = np.array([8.200e-03, 2.768e-02, 6.560e-02, 1.281e-01, 2.214e-01, 3.516e-01, 5.248e-01, 7.472e-01, 1.025e+00])

LN_T = np.log(T_TABLE)
LN_T_K = np.log(T_K_TABLE)
LN_SIGMA = np.log(SIGMA_TABLE)
LN_LAMBDA = np.log(LAMBDA_TABLE)
LN_CV = np.log(CV_TABLE)
LN_K = np.log(K_TABLE)


def interp_log(T, ln_T_table, ln_y_table):
    T = np.atleast_1d(T).astype(float)
    T_clipped = np.clip(T, np.exp(ln_T_table[0]), np.exp(ln_T_table[-1]))
    return np.exp(np.interp(np.log(T_clipped), ln_T_table, ln_y_table))


def sigma_T(T):
    return interp_log(T, LN_T, LN_SIGMA)


def lambda_T(T):
    return interp_log(T, LN_T, LN_LAMBDA)


def cv_T(T):
    return interp_log(T, LN_T, LN_CV)


def k_T(T):
    """k(T) с экстраполяцией за пределы таблицы"""
    T = np.atleast_1d(T).astype(float)
    result = np.zeros_like(T)
    for i, ti in enumerate(T):
        if ti <= T_K_TABLE[-1]:
            result[i] = interp_log(ti, LN_T_K, LN_K)
        else:
            # Экстраполяция: ln(k) = a + b * ln(T)
            lnT1, lnT2 = LN_T_K[-2], LN_T_K[-1]
            lnk1, lnk2 = LN_K[-2], LN_K[-1]
            b = (lnk2 - lnk1) / (lnT2 - lnT1)
            a = lnk2 - b * lnT2
            result[i] = np.exp(a + b * np.log(ti))
    return result


# Параметры
R = 0.35
T_0 = 8000.0
T_W = 1800.0
P = 2
I_MAX = 1000.0
t_MAX = 80e-6


def T_func(r, R=R, T_w=T_W, T_0=T_0, p=P):
    return T_0 + (T_w - T_0) * (r / R) ** p


def u_planck(T):
    T = np.atleast_1d(T).astype(float)
    return 3.084e-4 / (np.exp(4.799e4 / T) - 1.0)


def I_t(t, I_max=I_MAX, t_max=t_MAX):
    return I_max * (t / t_max) * np.exp(1.0 - t / t_max)


def harmonic(a, b):
    """Гармоническое среднее"""
    s = a + b
    return np.where(s > 1e-300, 2.0 * a * b / s, 0.0)


def thomas(A, B, C, D):
    """Метод прогонки для трёхдиагональной системы"""
    n = len(B)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    
    alpha[0] = -C[0] / B[0]
    beta[0] = D[0] / B[0]
    
    for i in range(1, n - 1):
        denom = B[i] + A[i] * alpha[i - 1]
        alpha[i] = -C[i] / denom
        beta[i] = (D[i] - A[i] * beta[i - 1]) / denom
    
    y = np.zeros(n)
    denom_last = B[-1] + A[-1] * alpha[-2]
    y[-1] = (D[-1] - A[-1] * beta[-2]) / denom_last
    
    for i in range(n - 2, -1, -1):
        y[i] = alpha[i] * y[i + 1] + beta[i]
    
    return y


# ============================================================
# Решение для u(r) — ИСПРАВЛЕНО
# ============================================================
def solve_radiation(r, T):
    N = len(r) - 1
    h = r[1] - r[0]
    R_max = r[-1]
    
    k_arr = k_T(T)
    up = u_planck(T)
    
    # χ = 1/(3k)
    chi = 1.0 / (3.0 * k_arr + 1e-300)
    
    # χ на гранях ячеек (гармоническое среднее)
    chi_iface = np.zeros(N)
    for i in range(N):
        chi_iface[i] = harmonic(chi[i], chi[i + 1])
    
    A = np.zeros(N + 1)
    B = np.zeros(N + 1)
    C = np.zeros(N + 1)
    D = np.zeros(N + 1)
    
    # ГУ при r = 0: du/dr = 0
    r_half = 0.5 * h
    V0 = 0.5 * r_half * r_half
    chi0 = chi_iface[0]
    A[0] = 0.0
    B[0] = r_half * chi0 / h + 3.0 * k_arr[0] * V0
    C[0] = -r_half * chi0 / h
    D[0] = 3.0 * k_arr[0] * up[0] * V0
    
    # Внутренние узлы
    for n in range(1, N):
        rn_minus = r[n] - 0.5 * h
        rn_plus = r[n] + 0.5 * h
        Vn = 0.5 * (rn_plus ** 2 - rn_minus ** 2)
        chi_m = chi_iface[n - 1]
        chi_p = chi_iface[n]
        A[n] = -rn_minus * chi_m / h
        C[n] = -rn_plus * chi_p / h
        B[n] = -A[n] - C[n] + 3.0 * k_arr[n] * Vn
        D[n] = 3.0 * k_arr[n] * up[n] * Vn
    
    # ГУ при r = R: -χ du/dr = 0.39 * u
    rN_minus = R_max - 0.5 * h
    VN = 0.5 * (R_max ** 2 - rN_minus ** 2)
    chi_last = chi_iface[N - 1]
    A[N] = -rN_minus * chi_last / h
    B[N] = rN_minus * chi_last / h + 3.0 * k_arr[N] * VN + R_max * 3.0 * 0.39
    C[N] = 0.0
    D[N] = 3.0 * k_arr[N] * up[N] * VN
    
    u = thomas(A, B, C, D)
    return u, k_arr, up


# ============================================================
# Напряжённость поля E
# ============================================================
def compute_E(r, T, I):
    integrand = sigma_T(T) * r
    integral = _trapz(integrand, r)
    return I / (2.0 * np.pi * integral) if integral > 0 else 0.0


# ============================================================
# Один шаг по времени — С ИТЕРАЦИЯМИ
# ============================================================
def step_T(r, T, tau, I_now, T_w, max_iter=30, eps=1e-4, omega=0.7):
    N = len(r) - 1
    h = r[1] - r[0]
    
    T_curr = T.copy()
    u_curr = None
    
    for it in range(max_iter):
        # Вычисляем коэффициенты по текущему T_curr
        sigma_a = sigma_T(T_curr)
        lam_a = lambda_T(T_curr)
        cv_a = cv_T(T_curr)
        k_a = k_T(T_curr)
        
        # Решаем уравнение переноса
        u_a, _, up_a = solve_radiation(r, T_curr)
        u_curr = u_a
        
        # Напряжённость поля
        E = compute_E(r, T_curr, I_now)
        
        # Теплопроводность на гранях (гармоническое среднее)
        lam_iface = np.zeros(N)
        for i in range(N):
            lam_iface[i] = harmonic(lam_a[i], lam_a[i + 1])
        
        A = np.zeros(N + 1)
        B = np.zeros(N + 1)
        C = np.zeros(N + 1)
        D = np.zeros(N + 1)
        
        # ГУ при r = 0: симметрия
        r_half = 0.5 * h
        V0 = 0.5 * r_half * r_half
        lam0 = lam_iface[0]
        A[0] = 0.0
        B[0] = cv_a[0] * V0 / tau + r_half * lam0 / h
        C[0] = -r_half * lam0 / h
        D[0] = (cv_a[0] * V0 / tau * T_curr[0] + 
                sigma_a[0] * E * E * V0 - 
                C_LIGHT * k_a[0] * (up_a[0] - u_a[0]) * V0)
        
        # Внутренние узлы
        for n in range(1, N):
            rn_minus = r[n] - 0.5 * h
            rn_plus = r[n] + 0.5 * h
            Vn = 0.5 * (rn_plus ** 2 - rn_minus ** 2)
            lam_m = lam_iface[n - 1]
            lam_p = lam_iface[n]
            A[n] = -rn_minus * lam_m / h
            C[n] = -rn_plus * lam_p / h
            B[n] = -A[n] - C[n] + cv_a[n] * Vn / tau
            D[n] = (cv_a[n] * Vn / tau * T_curr[n] + 
                    sigma_a[n] * E * E * Vn - 
                    C_LIGHT * k_a[n] * (up_a[n] - u_a[n]) * Vn)
        
        # ГУ при r = R: T = T_w
        A[N] = 0.0
        B[N] = 1.0
        C[N] = 0.0
        D[N] = T_w
        
        # Решаем систему
        T_pred = thomas(A, B, C, D)
        
        # Релаксация
        T_new = omega * T_pred + (1.0 - omega) * T_curr
        
        # Проверка сходимости
        rel_change = np.max(np.abs(T_new - T_curr)) / np.max(np.abs(T_new))
        if rel_change < eps:
            return T_new, E, u_curr, up_a
        
        T_curr = T_new
    
    return T_new, E, u_curr, up_a


# ============================================================
# Полный расчёт
# ============================================================
def solve(N=100, tau=1e-7, t_end=200e-6, verbose=False):
    r = np.linspace(0.0, R, N + 1)
    T = T_func(r)
    
    n_steps = int(round(t_end / tau))
    t_arr = np.zeros(n_steps + 1)
    T_hist = np.zeros((n_steps + 1, N + 1))
    I_hist = np.zeros(n_steps + 1)
    E_hist = np.zeros(n_steps + 1)
    
    T_hist[0] = T.copy()
    I_hist[0] = I_t(0.0)
    E_hist[0] = 0.0
    
    for n in range(n_steps):
        t_new = (n + 1) * tau
        I_new = I_t(t_new)
        T, E, _, _ = step_T(r, T, tau, I_new, T_W)
        t_arr[n + 1] = t_new
        T_hist[n + 1] = T.copy()
        I_hist[n + 1] = I_new
        E_hist[n + 1] = E
        
        if verbose and (n + 1) % max(1, n_steps // 10) == 0:
            print(f"  t = {t_new * 1e6:7.2f} мкс,  T(0) = {T[0]:7.1f} К,  "
                  f"T(R/2) = {T[N // 2]:7.1f} К,  I = {I_new:7.1f} А,  E = {E:7.2f} В/см")
    
    return r, t_arr, T_hist, I_hist, E_hist


# ============================================================
# Графики
# ============================================================
def plot_T_profiles(r, t_arr, T_hist, t_max_pulse, save_times=None, fname='lab4_profiles.png'):
    if save_times is None:
        save_times = list(np.linspace(t_max_pulse * 0.1, t_max_pulse, 5)) + \
                     list(np.linspace(t_max_pulse * 1.2, t_arr[-1], 5))
    
    plt.figure(figsize=(11, 6))
    for ts in save_times:
        idx = int(np.argmin(np.abs(t_arr - ts)))
        ts_actual = t_arr[idx]
        ls = '-' if ts_actual <= t_max_pulse else '--'
        plt.plot(r, T_hist[idx], ls, label=f't = {ts_actual * 1e6:5.1f} мкс')
    plt.xlabel('r, см')
    plt.ylabel('T(r,t), К')
    plt.title('Профили температуры на переднем (—) и заднем (--) фронтах')
    plt.grid(True, alpha=0.4)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_dynamics(r, t_arr, T_hist, I_hist, E_hist, fname='lab4_dynamics.png'):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    
    ax = axes[0, 0]
    ax.plot(t_arr * 1e6, T_hist[:, 0], 'b', lw=2, label='T(0,t)')
    ax.plot(t_arr * 1e6, T_hist[:, len(r) // 2], 'g', lw=1.5, label='T(R/2,t)')
    ax.set_xlabel('t, мкс')
    ax.set_ylabel('T, К')
    ax.set_title('Температура во времени')
    ax.grid(True, alpha=0.4)
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(t_arr * 1e6, I_hist, 'r', lw=2)
    ax.set_xlabel('t, мкс')
    ax.set_ylabel('I(t), А')
    ax.set_title('Импульс тока')
    ax.grid(True, alpha=0.4)
    
    ax = axes[1, 0]
    ax.plot(t_arr * 1e6, E_hist, 'm', lw=2)
    ax.set_xlabel('t, мкс')
    ax.set_ylabel('E(t), В/см')
    ax.set_title('Напряжённость поля')
    ax.grid(True, alpha=0.4)
    
    ax = axes[1, 1]
    P = E_hist * I_hist
    ax.plot(t_arr * 1e6, P, 'k', lw=2)
    ax.set_xlabel('t, мкс')
    ax.set_ylabel('E·I, Вт/см')
    ax.set_title('Подводимая мощность')
    ax.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ============================================================
# Краткая таблица первых значений (t=0 и первый шаг)
# ============================================================

def print_short_table():
    print("\n" + "=" * 70)
    print("КРАТКАЯ ТАБЛИЦА ПЕРВЫХ ЗНАЧЕНИЙ")
    print("=" * 70)
    
    N = 100
    tau = 1e-7
    r = np.linspace(0.0, R, N + 1)
    
    # ========== 1. Начальный момент t = 0 ==========
    T0 = T_func(r)
    I0 = I_t(0.0)
    E0 = 0.0  # при t=0 ток=0 → E=0
    sigma0 = sigma_T(T0)
    sigma_E2_0 = sigma0 * E0**2  # = 0
    
    # Для div_rad при t=0: считаем u при начальной T
    u0, _, up0 = solve_radiation(r, T0)
    div_rad_0 = C_LIGHT * k_T(T0) * (up0 - u0)
    
    print("\n--- t = 0 (начальный момент) ---")
    print(f"Ток I = {I0:.4f} А,  E = {E0:.6f} В/см\n")
    print(f"{'r, см':>8} {'T, К':>12} {'σE², Вт/см³':>18} {'div_rad, Вт/см³':>20}")
    print("-" * 62)
    
    for i in range(0, N+1, N//10):
        print(f"{r[i]:8.4f} {T0[i]:12.2f} {sigma_E2_0[i]:18.6e} {div_rad_0[i]:20.6e}")
    
    # ========== 2. Первый шаг t = tau = 0.1 мкс ==========
    print("\n--- Первый шаг по времени (t = tau = 0.1 мкс) ---")
    
    # Получаем решение на первом шаге
    _, t_arr, T_hist, I_hist, E_hist = solve(N=N, tau=tau, t_end=tau, verbose=False)
    
    T1 = T_hist[1]
    I1 = I_hist[1]
    E1 = E_hist[1]
    
    sigma1 = sigma_T(T1)
    sigma_E2_1 = sigma1 * E1**2
    
    u1, _, up1 = solve_radiation(r, T1)
    div_rad_1 = C_LIGHT * k_T(T1) * (up1 - u1)
    
    print(f"Время: t = {tau*1e6:.4f} мкс")
    print(f"Ток I = {I1:.4f} А")
    print(f"Напряжённость E = {E1:.6f} В/см\n")
    print(f"{'r, см':>8} {'T, К':>12} {'σE², Вт/см³':>18} {'div_rad, Вт/см³':>20}")
    print("-" * 62)
    
    for i in range(0, N+1, N//10):
        print(f"{r[i]:8.4f} {T1[i]:12.2f} {sigma_E2_1[i]:18.6e} {div_rad_1[i]:20.6e}")
    
    print("\n" + "=" * 70)

# Вызов функции (раскомментировать при необходимости)
# print_short_table()

if __name__ == '__main__':
    print("Лабораторная работа №4")
    print(f"Параметры: T0 = {T_0} К, Tw = {T_W} К, R = {R} см, ")
    print(f"I_max = {I_MAX} А, t_max = {t_MAX*1e6:.0f} мкс\n")
    print_short_table()
    
    # Основной расчёт
    r, t_arr, T_hist, I_hist, E_hist = solve(N=100, tau=1e-7, t_end=240e-6, verbose=True)
    
    # Графики
    save_times = [10e-6, 30e-6, 50e-6, 70e-6, 80e-6, 100e-6, 130e-6, 160e-6, 200e-6, 240e-6]
    plot_T_profiles(r, t_arr, T_hist, t_max_pulse=80e-6, save_times=save_times)
    plot_dynamics(r, t_arr, T_hist, I_hist, E_hist)
    
    print("\nГотово. Графики сохранены.")