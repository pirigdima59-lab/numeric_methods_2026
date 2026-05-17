import numpy as np
import matplotlib.pyplot as plt


# ТЕСТОВЕ ДИФЕРЕНЦІАЛЬНЕ РІВНЯННЯ ТА ТОЧНИЙ РОЗВ'ЯЗОК
def f(x, y):
    return x - y


# Аналітичний розв'язок
def y_exact(x):
    return x - 1 + 2 * np.exp(-x)


# ЧАСТИНА 1 МЕТОД АДАМСА ПРОГНОЗ КОРЕКЦІЯ 2 ГО ПОРЯДКУ
def adams_method_auto_step(x0, y0, x_end, h0, eps):
    x_vals = [x0]
    y_vals = [y0]
    h_vals = [h0]

    # Знаходження додаткової точки методом Рунге Кутта 4 го порядку
    h = h0
    k1 = f(x0, y0)
    k2 = f(x0 + h / 2, y0 + h * k1 / 2)
    k3 = f(x0 + h / 2, y0 + h * k2 / 2)
    k4 = f(x0 + h, y0 + h * k3)
    y1 = y0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    x_vals.append(x0 + h)
    y_vals.append(y1)
    h_vals.append(h)

    err_exact = [0, abs(y1 - y_exact(x0 + h))]
    err_est = [0, 0]

    x_n = x_vals[-1]

    while x_n < x_end:
        if x_n + h > x_end:
            h = x_end - x_n

        y_n = y_vals[-1]
        y_n_minus_1 = y_vals[-2]

        fn = f(x_n, y_n)
        fn_minus_1 = f(x_vals[-2], y_n_minus_1)

        # Етап прогнозу
        y_pr = y_n + (h / 2) * (3 * fn - fn_minus_1)

        # Етап корекції
        fn_plus_1_pr = f(x_n + h, y_pr)
        y_cor = y_n + (h / 2) * (fn_plus_1_pr + fn)

        # Оцінка локальної похибки
        local_err_est = abs(y_cor - y_pr) / 6.0

        # Автоматичний вибір кроку
        if local_err_est > eps:
            h /= 2.0
            continue

        x_n += h
        x_vals.append(x_n)
        y_vals.append(y_cor)

        err_exact.append(abs(y_cor - y_exact(x_n)))
        err_est.append(local_err_est)
        h_vals.append(h)

        if local_err_est < eps / 8.0:
            h *= 2.0

    return np.array(x_vals), np.array(y_vals), np.array(h_vals), np.array(err_exact), np.array(err_est)


# ЧАСТИНА 2 МЕТОД РУНГЕ КУТТА 4 ГО ПОРЯДКУ
def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_auto_step(x0, y0, x_end, h0, eps):
    x_vals = [x0]
    y_vals = [y0]
    h_vals = [h0]

    err_exact = [0]
    err_runge = [0]

    x_n = x0
    y_n = y0
    h = h0

    while x_n < x_end:
        if x_n + h > x_end:
            h = x_end - x_n

        y_full_step = rk4_step(x_n, y_n, h)

        y_half_step_1 = rk4_step(x_n, y_n, h / 2)
        y_half_step_2 = rk4_step(x_n + h / 2, y_half_step_1, h / 2)

        # Оцінка похибки за правилом Рунге згідно з методичкою
        local_err_est = (16 / 15) * abs(y_half_step_2 - y_full_step)

        if local_err_est > eps:
            h /= 2.0
            continue

        x_n += h
        y_n = y_half_step_2

        x_vals.append(x_n)
        y_vals.append(y_n)
        h_vals.append(h)

        err_exact.append(abs(y_n - y_exact(x_n)))
        err_runge.append(local_err_est)

        if local_err_est < eps / 32.0:
            h *= 2.0

    return np.array(x_vals), np.array(y_vals), np.array(h_vals), np.array(err_exact), np.array(err_runge)


# ГОЛОВНИЙ БЛОК ВИКОНАННЯ ТА ВИВІД РЕЗУЛЬТАТІВ
if __name__ == "__main__":
    x0, y0 = 0.0, 1.0
    x_end = 2.0
    h_initial = 0.1
    tolerance = 1e-5

    x_adams, y_adams, h_adams, err_ex_adams, err_est_adams = adams_method_auto_step(x0, y0, x_end, h_initial, tolerance)
    x_rk, y_rk, h_rk, err_ex_rk, err_est_rk = rk4_auto_step(x0, y0, x_end, h_initial, tolerance)

    # Вивід результатів у консоль
    print("РОЗВ'ЯЗОК ЗАДАЧІ КОШІ З АВТОМАТИЧНИМ КРОКОМ")
    print(f"Точність розв'язку: {tolerance}\n")

    print("МЕТОД АДАМСА ПРОГНОЗ КОРЕКЦІЯ")
    print("   x     |     y(x)   |   Крок h   | Локальна похибка")
    for i in range(len(x_adams)):
        # Виводимо лише кожну 5 ту точку щоб не засмічувати консоль
        if i % 5 == 0 or i == len(x_adams) - 1:
            print(f"{x_adams[i]:8.4f} | {y_adams[i]:10.6f} | {h_adams[i]:10.6f} | {err_est_adams[i]:.4e}")

    print(f"Всього кроків Адамса: {len(x_adams) - 1}\n")

    print("МЕТОД РУНГЕ КУТТА 4 ГО ПОРЯДКУ")
    print("   x     |     y(x)   |   Крок h   | Локальна похибка")
    for i in range(len(x_rk)):
        if i % 5 == 0 or i == len(x_rk) - 1:
            print(f"{x_rk[i]:8.4f} | {y_rk[i]:10.6f} | {h_rk[i]:10.6f} | {err_est_rk[i]:.4e}")

    print(f"Всього кроків Рунге Кутта: {len(x_rk) - 1}\n")

    # Побудова графіків
    plt.figure(figsize=(12, 10))
    plt.suptitle("Частина 1 Метод Адамса", fontsize=14)

    plt.subplot(3, 1, 1)
    plt.plot(x_adams, y_exact(x_adams), 'k-', label="Точний розв'язок")
    plt.plot(x_adams, y_adams, 'r--', label="Метод Адамса")
    plt.title("Розв'язок")
    plt.grid(True);
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(x_adams, err_ex_adams, 'b-', label="Справжня похибка")
    plt.plot(x_adams, err_est_adams, 'r--', label="Оцінка похибки")
    plt.title("Локальна похибка")
    plt.grid(True);
    plt.legend();
    plt.yscale('log')

    plt.subplot(3, 1, 3)
    plt.step(x_adams, h_adams, 'g-', where='post', label="Крок h(x)")
    plt.title("Залежність величини кроку від x")
    plt.grid(True);
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("lab10_adams.png")
    print("Графіки для методу Адамса збережено у файл lab10_adams.png")

    plt.figure(figsize=(12, 10))
    plt.suptitle("Частина 2 Метод Рунге Кутта", fontsize=14)

    plt.subplot(3, 1, 1)
    plt.plot(x_rk, y_exact(x_rk), 'k-', label="Точний розв'язок")
    plt.plot(x_rk, y_rk, 'b--', label="Метод Рунге Кутта")
    plt.title("Розв'язок")
    plt.grid(True);
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(x_rk, err_ex_rk, 'b-', label="Справжня похибка")
    plt.plot(x_rk, err_est_rk, 'r--', label="Оцінка за правилом Рунге")
    plt.title("Локальна похибка")
    plt.grid(True);
    plt.legend();
    plt.yscale('log')

    plt.subplot(3, 1, 3)
    plt.step(x_rk, h_rk, 'g-', where='post', label="Крок h(x)")
    plt.title("Залежність величини кроку від x")
    plt.grid(True);
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("lab10_rungekutta.png")
    print("Графіки для методу Рунге Кутта збережено у файл lab10_rungekutta.png")