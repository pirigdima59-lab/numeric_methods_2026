import numpy as np
import matplotlib.pyplot as plt
import math
import cmath



# ЧАСТИНА 1: ТРАНСЦЕНДЕНТНІ РІВНЯННЯ

# Досліджувана функція: F(x) = x^3 - 4x + 1
# 1) на відрізку [0, 1] (функція спадає)
# 2) на відрізку [1.5, 2] (функція зростає)

def f(x): return x ** 3 - 4 * x + 1


def df(x): return 3 * x ** 2 - 4


def d2f(x): return 6 * x


# Спільний критерій зупинки: |F(x_{n+1})| < eps ТА |x_{n+1} - x_n| < eps
def check_stop(xn, xn1, eps=1e-10):
    return abs(f(xn1)) < eps and abs(xn1 - xn) < eps


# 1. Табуляція та графік
def part1_tabulate_and_plot(a, b, h):
    x_vals = np.arange(a, b + h, h)
    y_vals = f(x_vals)

    # Запис у текстовий файл
    with open("lab8_tabulation.txt", "w", encoding="utf-8") as file:
        file.write("Табуляція функції F(x) = x^3 - 4x + 1\n")
        file.write("-" * 30 + "\n")
        for x, y in zip(x_vals, y_vals):
            file.write(f"x: {x:6.2f} | F(x): {y:8.4f}\n")

    # Побудова графіка
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="F(x) = x³ - 4x + 1", color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Виділення коренів трансцендентного рівняння")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lab8_plot.png")
    print("Графік збережено у файл 'lab8_plot.png'")
    print("Табуляцію збережено у файл 'lab8_tabulation.txt'\n")


#  Ітераційні методи
def simple_iteration(x0, tau, eps=1e-10, max_iter=1000):
    xn = x0
    for i in range(max_iter):
        xn1 = xn + tau * f(xn)
        if check_stop(xn, xn1, eps): return xn1, i + 1
        xn = xn1
    return xn, max_iter


def newton_method(x0, eps=1e-10, max_iter=1000):
    xn = x0
    for i in range(max_iter):
        if df(xn) == 0: break
        xn1 = xn - f(xn) / df(xn)
        if check_stop(xn, xn1, eps): return xn1, i + 1
        xn = xn1
    return xn, max_iter


def chebyshev_method(x0, eps=1e-10, max_iter=1000):
    xn = x0
    for i in range(max_iter):
        if df(xn) == 0: break
        term1 = f(xn) / df(xn)
        term2 = 0.5 * (f(xn) ** 2 * d2f(xn)) / (df(xn) ** 3)
        xn1 = xn - term1 - term2
        if check_stop(xn, xn1, eps): return xn1, i + 1
        xn = xn1
    return xn, max_iter


def secant_method(x0, x1, eps=1e-10, max_iter=1000):
    xn_minus_1, xn = x0, x1
    for i in range(max_iter):
        if f(xn) - f(xn_minus_1) == 0: break
        xn1 = xn - f(xn) * (xn - xn_minus_1) / (f(xn) - f(xn_minus_1))
        if check_stop(xn, xn1, eps): return xn1, i + 1
        xn_minus_1, xn = xn, xn1
    return xn, max_iter


def parabola_method(x0, x1, x2, eps=1e-10, max_iter=1000):
    xn2, xn1, xn = x0, x1, x2
    for i in range(max_iter):
        h1, h2 = xn1 - xn2, xn - xn1
        d1 = (f(xn1) - f(xn2)) / h1
        d2 = (f(xn) - f(xn1)) / h2

        A = (d2 - d1) / (h2 + h1)
        B = d2 + h2 * A
        C = f(xn)

        sign = 1 if B > 0 else -1
        denom = B + sign * np.sqrt(abs(B ** 2 - 4 * A * C))
        if denom == 0: break

        dx = -2 * C / denom
        xn1_new = xn + dx

        if check_stop(xn, xn1_new, eps): return xn1_new, i + 1
        xn2, xn1, xn = xn1, xn, xn1_new
    return xn, max_iter


def inverse_interpolation(x0, x1, x2, eps=1e-10, max_iter=1000):
    xn2, xn1, xn = x0, x1, x2
    for i in range(max_iter):
        f0, f1, f2 = f(xn2), f(xn1), f(xn)
        try:
            L0 = xn2 * (f1 * f2) / ((f0 - f1) * (f0 - f2))
            L1 = xn1 * (f0 * f2) / ((f1 - f0) * (f1 - f2))
            L2 = xn * (f0 * f1) / ((f2 - f0) * (f2 - f1))
            xn1_new = L0 + L1 + L2
        except ZeroDivisionError:
            break

        if check_stop(xn, xn1_new, eps): return xn1_new, i + 1
        xn2, xn1, xn = xn1, xn, xn1_new
    return xn, max_iter



# ЧАСТИНА 2: АЛГЕБРАЇЧНІ РІВНЯННЯ

# Алгебраїчне рівняння: P(x) = -2 + x - 2x^2 + x^3 = 0

def write_poly(filename, coeffs):
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(" ".join(map(str, coeffs)))


def read_poly(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        return list(map(float, f.read().split()))


def horner_eval(coeffs, x):
    m = len(coeffs) - 1
    b = coeffs[m]
    for i in range(m - 1, -1, -1):
        b = coeffs[i] + x * b
    return b


def newton_horner(coeffs, x0, eps=1e-10, max_iter=1000):
    m = len(coeffs) - 1
    xn = x0
    for it in range(max_iter):
        b = [0] * (m + 1)
        b[m] = coeffs[m]
        for i in range(m - 1, -1, -1):
            b[i] = coeffs[i] + xn * b[i + 1]

        c = [0] * (m + 1)
        c[m] = b[m]
        for i in range(m - 1, 0, -1):
            c[i] = b[i] + xn * c[i + 1]

        fxn = b[0]
        dfxn = c[1]

        if dfxn == 0: break
        xn1 = xn - fxn / dfxn

        if abs(fxn) < eps and abs(xn1 - xn) < eps:
            return xn1, it + 1
        xn = xn1
    return xn, max_iter


def lin_method(coeffs, p0, q0, eps=1e-10, max_iter=1000):
    m = len(coeffs) - 1
    p, q = p0, q0

    for it in range(max_iter):
        b = [0] * (m + 1)
        b[m] = coeffs[m]
        b[m - 1] = coeffs[m - 1] - p * b[m]

        for i in range(m - 2, -1, -1):
            b[i] = coeffs[i] - p * b[i + 1] - q * b[i + 2]

        if b[2] == 0: break

        p_new = (coeffs[1] - q * b[3]) / b[2] if m >= 3 else coeffs[1] / b[2]
        q_new = coeffs[0] / b[2]

        if abs(p_new - p) < eps and abs(q_new - q) < eps:
            p, q = p_new, q_new
            break
        p, q = p_new, q_new

    alpha = -p / 2
    disc = q - alpha ** 2
    if disc > 0:
        beta = math.sqrt(disc)
        root1 = complex(alpha, beta)
        root2 = complex(alpha, -beta)
    else:
        beta = math.sqrt(-disc)
        root1 = complex(alpha + beta, 0)
        root2 = complex(alpha - beta, 0)

    return root1, root2, it + 1


# ГОЛОВНИЙ БЛОК ВИКОНАННЯ

if __name__ == "__main__":
    print("=" * 50)
    print("ЧАСТИНА 1: ЧИСЕЛЬНІ МЕТОДИ (ТРАНСЦЕНДЕНТНІ РІВНЯННЯ)")
    print("=" * 50)

    # 1. Табуляція
    part1_tabulate_and_plot(-2, 2, 0.1)

    # 2. Розв'язок
    roots_to_test = [
        {"name": "КОРІНЬ 1 (спадання, [0, 1])", "x0": 0.5, "x1": 0.4, "x2": 0.6, "tau": 0.2},
        {"name": "КОРІНЬ 2 (зростання, [1.5, 2])", "x0": 1.8, "x1": 1.7, "x2": 1.9, "tau": -0.1}
    ]

    for rt in roots_to_test:
        print(f"\n--- {rt['name']} ---")
        x, it = simple_iteration(rt['x0'], rt['tau'])
        print(f"Метод простої ітерації:    x = {x:.10f} | ітерацій = {it}")

        x, it = newton_method(rt['x0'])
        print(f"Метод Ньютона:             x = {x:.10f} | ітерацій = {it}")

        x, it = chebyshev_method(rt['x0'])
        print(f"Метод Чебишева:            x = {x:.10f} | ітерацій = {it}")

        x, it = secant_method(rt['x0'], rt['x1'])
        print(f"Метод хорд:                x = {x:.10f} | ітерацій = {it}")

        x, it = parabola_method(rt['x0'], rt['x1'], rt['x2'])
        print(f"Метод парабол:             x = {x:.10f} | ітерацій = {it}")

        x, it = inverse_interpolation(rt['x0'], rt['x1'], rt['x2'])
        print(f"Зворотня інтерполяція:     x = {x:.10f} | ітерацій = {it}")

    print("\n\n" + "=" * 50)
    print("ЧАСТИНА 2: АЛГЕБРАЇЧНІ РІВНЯННЯ")
    print("=" * 50)

    poly_file = "lab8_poly_coeffs.txt"
    # Рівняння: -2 + 1*x - 2*x^2 + 1*x^3 = 0
    # Має корінь x=2 і комплексно-спряжені корені i, -i
    initial_coeffs = [-2.0, 1.0, -2.0, 1.0]

    write_poly(poly_file, initial_coeffs)
    loaded_coeffs = read_poly(poly_file)

    print(f"Коефіцієнти полінома зчитано з файлу '{poly_file}': {loaded_coeffs}")
    print(f"Перевірка схеми Горнера (P(2) має бути 0): P(2) = {horner_eval(loaded_coeffs, 2.0)}\n")

    # Дійсний корінь
    real_root, iters_n = newton_horner(loaded_coeffs, x0=1.5)
    print("--- Метод Ньютона-Горнера (Дійсний корінь) ---")
    print(f"Корінь = {real_root:.10f} | Ітерацій = {iters_n}\n")

    # Комплексні корені
    # Початкове наближення для дільника x^2 + px + q (візьмемо p0=0, q0=0.5)
    c_root1, c_root2, iters_l = lin_method(loaded_coeffs, p0=0.0, q0=0.5)
    print("--- Метод Ліна (Комплексні корені) ---")
    print(f"Корінь 1 = {c_root1.real:.10f} + {c_root1.imag:.10f}j")
    print(f"Корінь 2 = {c_root2.real:.10f} + {c_root2.imag:.10f}j")
    print(f"Ітерацій = {iters_l}")