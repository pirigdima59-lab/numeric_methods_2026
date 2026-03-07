import csv
import math
import matplotlib.pyplot as plt

# 1. Дані та CSV

X = [100, 200, 400, 800, 1600]
Y = [120, 110,  90,  65,   40]

def write_csv(fname, x, y):
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["objects", "fps"])
        w.writeheader()
        for xi, yi in zip(x, y):
            w.writerow({"objects": xi, "fps": yi})

def read_csv(fname):
    x, y = [], []
    with open(fname, newline="") as f:
        for row in csv.DictReader(f):
            x.append(float(row["objects"]))
            y.append(float(row["fps"]))
    return x, y

# 2. Розділені різниці

def build_div_diff(x, y):
    n = len(x)
    T = [list(y)]
    for k in range(1, n):
        row = [(T[k-1][i+1] - T[k-1][i]) / (x[i+k] - x[i]) for i in range(n-k)]
        T.append(row)
    return T

def print_div_diff(x, T):
    n = len(x)
    header = f"{'i':>3} {'x_i':>7} " + "".join(f"  f[x_i..+{k}]" for k in range(n))
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{i:>3} {x[i]:>7.0f} "
        row += "".join(f"  {T[k][i]:>12.6f}" for k in range(n - i))
        print(row)

# 3. Метод Ньютона

def newton(x, y, t):
    T = build_div_diff(x, y)
    result = T[0][0]
    product = 1.0
    for k in range(1, len(x)):
        product *= (t - x[k-1])
        result  += T[k][0] * product
    return result

# 4. Факторіальні многочлени

def factorial_poly(x, y, t):
    n = len(x)
    s = (t - x[0]) / (x[-1] - x[0]) * (n - 1)
    d = list(y)
    diffs = [d[0]]
    for _ in range(n - 1):
        d = [d[i+1] - d[i] for i in range(len(d) - 1)]
        diffs.append(d[0])
    result = 0.0
    binom  = 1.0
    for k in range(n):
        if k > 0:
            binom *= (s - k + 1) / k
        result += binom * diffs[k]
    return result

# 5. Метод Лагранжа

def lagrange(x, y, t):
    n = len(x)
    result = 0.0
    for i in range(n):
        basis = 1.0
        for j in range(n):
            if j != i:
                basis *= (t - x[j]) / (x[i] - x[j])
        result += y[i] * basis
    return result

# 6. Графіки

def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]

def plot_results(x, y):
    xd = linspace(x[0], x[-1], 300)

    yn = [newton(x, y, t)         for t in xd]
    yf = [factorial_poly(x, y, t) for t in xd]
    yl = [lagrange(x, y, t)       for t in xd]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Варіант 5 — Прогнозування FPS", fontsize=13, fontweight="bold")

    # Графік 1: усі три методи
    ax = axes[0][0]
    ax.plot(xd, yn, "b-",  lw=2, label="Ньютон")
    ax.plot(xd, yf, "g--", lw=2, label="Факторіальний")
    ax.plot(xd, yl, "m:",  lw=2, label="Лагранж")
    ax.scatter(x, y, color="red", zorder=5, s=60, label="Вузли")
    ax.axvline(1000, color="purple", ls=":",  lw=1.5, label="x = 1000")
    ax.axhline(60,   color="orange", ls="--", lw=1,   label="FPS = 60")
    ax.set_xlabel("Кількість об'єктів")
    ax.set_ylabel("FPS")
    ax.set_title("Інтерполяційні криві")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Графік 2: похибка між методами при n = 5, 10, 20
    ax2 = axes[0][1]
    for n_pts, col in [(5, "blue"), (10, "green"), (20, "red")]:
        xi = linspace(x[0], x[-1], n_pts)
        yi = [newton(x, y, xi_) for xi_ in xi]
        xd2 = linspace(xi[0], xi[-1], 300)
        err = [abs(newton(xi, yi, t) - factorial_poly(xi, yi, t)) for t in xd2]
        ax2.plot(xd2, err, color=col, lw=2, label=f"n = {n_pts}")
    ax2.set_xlabel("Кількість об'єктів")
    ax2.set_ylabel("|Ньютон − Факторіальний|")
    ax2.set_title("Похибка між методами при різних n")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Графік 3: ефект Рунге
    ax3 = axes[1][0]
    for n_pts, col in [(5, "blue"), (10, "green"), (20, "red")]:
        xi = linspace(x[0], x[-1], n_pts)
        yi = [newton(x, y, xi_) for xi_ in xi]
        xd3 = linspace(xi[0], xi[-1], 300)
        en = [abs(newton(xi, yi, t) - newton(x, y, t)) for t in xd3]
        ax3.plot(xd3, en, color=col, lw=2, label=f"n = {n_pts}")
    ax3.set_xlabel("Кількість об'єктів")
    ax3.set_ylabel("|P_n(x) − P_5(x)|")
    ax3.set_title("Ефект Рунге (відхилення від n=5)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Графік 4: порівняння з Лагранжем
    ax4 = axes[1][1]
    diff_nl = [abs(newton(x, y, t) - lagrange(x, y, t)) for t in xd]
    diff_fl = [abs(factorial_poly(x, y, t) - lagrange(x, y, t)) for t in xd]
    ax4.plot(xd, diff_nl, "b-",  lw=2, label="|Ньютон − Лагранж|")
    ax4.plot(xd, diff_fl, "g--", lw=2, label="|Факторіальний − Лагранж|")
    ax4.set_xlabel("Кількість об'єктів")
    ax4.set_ylabel("Різниця")
    ax4.set_title("Порівняння методів з Лагранжем")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("lab2_graphs.png", dpi=150)
    plt.close()
    print("Графік збережено: lab2_graphs.png")

# Головна

write_csv("fps_data.csv", X, Y)
x, y = read_csv("fps_data.csv")

print("Таблиця розділених різниць:")
T = build_div_diff(x, y)
print_div_diff(x, T)

t_pred = 1000.0
vn = newton(x, y, t_pred)
vf = factorial_poly(x, y, t_pred)
vl = lagrange(x, y, t_pred)
print(f"\nПрогноз FPS для {t_pred:.0f} об'єктів:")
print(f"  Метод Ньютона:           {vn:.4f}")
print(f"  Факторіальний многочлен: {vf:.4f}")
print(f"  Метод Лагранжа:          {vl:.4f}")

lo, hi = x[0], x[-1]
for _ in range(60):
    mid = (lo + hi) / 2
    if newton(x, y, mid) >= 60:
        lo = mid
    else:
        hi = mid
print(f"\nFPS >= 60 при кількості об'єктів <= {lo:.0f}")

plot_results(x, y)