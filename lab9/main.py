import numpy as np
import matplotlib.pyplot as plt


# 1. ТЕСТОВА ФУНКЦІЯ
# Функція Розенброка
def rosenbrock(X):
    x1, x2 = X[0], X[1]
    return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2


# 2. СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ ТА ЦІЛЬОВА ФУНКЦІЯ
# Задаємо систему нелінійних рівнянь
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2 - 4


def f2(x1, x2):
    return x2 - x1 ** 2


# Цільова функція як сума квадратів
def target_function_system(X):
    x1, x2 = X[0], X[1]
    return f1(x1, x2) ** 2 + f2(x1, x2) ** 2


# Побудова графіка системи рівнянь
def plot_system():
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X1, X2 = np.meshgrid(x, y)

    F1 = f1(X1, X2)
    F2 = f2(X1, X2)

    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, F1, levels=[0], colors='blue', linewidths=2)
    plt.contour(X1, X2, F2, levels=[0], colors='red', linewidths=2)

    import matplotlib.patches as mpatches
    blue_line = mpatches.Patch(color='blue', label='x1^2 + x2^2 - 4')
    red_line = mpatches.Patch(color='red', label='x2 - x1^2')
    plt.legend(handles=[blue_line, red_line])

    plt.title("Графіки системи нелінійних рівнянь")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.savefig("lab9_system_plot.png")
    print("Графік системи рівнянь збережено як lab9_system_plot.png")


# 3. АЛГОРИТМ ХУКА-ДЖИВСА

def exploratory_search(x_base, delta, func, reduce_step, q):
    # Досліджуючий пошук
    x = np.copy(x_base)
    f_base = func(x)
    improved = False

    for i in range(len(x)):
        # Крок вперед
        x[i] += delta[i]
        if func(x) < f_base:
            f_base = func(x)
            improved = True
        else:
            # Крок назад
            x[i] -= 2 * delta[i]
            if func(x) < f_base:
                f_base = func(x)
                improved = True
            else:
                # Повертаємося до попереднього стану
                x[i] += delta[i]
                if reduce_step:
                    delta[i] /= q

    return x, delta, improved


def hooke_jeeves(func, x0, delta0, q=2.0, p=1.0, eps1=1e-6, eps2=1e-6, max_iter=2000):
    # Головний цикл методу Хука-Дживса
    x_base = np.array(x0, dtype=float)
    delta = np.array(delta0, dtype=float)
    trajectory = [np.copy(x_base)]

    for iteration in range(max_iter):
        # Етап 1: Досліджуючий пошук
        x_new, delta, _ = exploratory_search(x_base, delta, func, reduce_step=True, q=q)

        if np.max(delta) < eps1 or func(x_new) < eps2:
            x_base = x_new
            trajectory.append(np.copy(x_base))
            break

        if np.array_equal(x_new, x_base):
            continue

        # Етап 2: Пошук по зразку
        while True:
            x_pattern = x_new + p * (x_new - x_base)

            x_pattern_exp, _, _ = exploratory_search(x_pattern, delta, func, reduce_step=False, q=q)

            if func(x_pattern_exp) < func(x_new):
                x_base = np.copy(x_new)
                x_new = np.copy(x_pattern_exp)
                trajectory.append(np.copy(x_base))
            else:
                x_base = np.copy(x_new)
                trajectory.append(np.copy(x_base))
                break

    return x_base, trajectory


# ГОЛОВНИЙ БЛОК ВИКОНАННЯ

if __name__ == "__main__":
    print("ЛАБОРАТОРНА РОБОТА №9: МЕТОД ХУКА-ДЖИВСА\n")

    # Побудова графіка
    plot_system()

    # Тестування програми на функції Розенброка
    print("\n[ТЕСТ] Оптимізація функції Розенброка")
    x0_rosen = [-1.2, 0.0]
    delta0_rosen = [0.5, 0.5]
    res_rosen, traj_rosen = hooke_jeeves(
        func=rosenbrock, x0=x0_rosen, delta0=delta0_rosen,
        q=2.0, p=1.0, eps1=1e-6, eps2=1e-6
    )
    print(f"Початкова точка: {x0_rosen}")
    print(f"Точка мінімуму: [{res_rosen[0]:.6f}, {res_rosen[1]:.6f}]")
    print(f"Мінімум функції: {rosenbrock(res_rosen):.8f}")
    print(f"Кількість кроків траєкторії: {len(traj_rosen)}")

    # Розв'язок системи нелінійних рівнянь
    print("\n[ЗАВДАННЯ] Розв'язок заданої системи нелінійних рівнянь")
    x0_sys = [1.0, 1.0]
    delta0_sys = [0.5, 0.5]

    res_sys, traj_sys = hooke_jeeves(
        func=target_function_system, x0=x0_sys, delta0=delta0_sys,
        q=2.0, p=1.0, eps1=1e-6, eps2=1e-6
    )

    print(f"Початкове наближення: {x0_sys}")
    print(f"Уточнений розв'язок: [{res_sys[0]:.6f}, {res_sys[1]:.6f}]")
    print(f"Значення Ф(X): {target_function_system(res_sys):.10f}")
    print(f"Нев'язка рівняння 1: {f1(res_sys[0], res_sys[1]):.8f}")
    print(f"Нев'язка рівняння 2: {f2(res_sys[0], res_sys[1]):.8f}")
    print(f"Кількість кроків: {len(traj_sys)}")

    # Виведення координат траєкторії у файл
    filename = "lab9_trajectory.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Траєкторія спуску для системи рівнянь\n")
        f.write(f"Початкова точка: {x0_sys}\n\n")
        f.write("Крок |     x1     |     x2     |    Ф(X)\n\n")
        for i, pt in enumerate(traj_sys):
            f.write(f"{i:4d} | {pt[0]:10.6f} | {pt[1]:10.6f} | {target_function_system(pt):.8e}\n")

    print(f"\nКоординати точок траєкторії спуску записано у файл {filename}")