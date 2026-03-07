import csv
import matplotlib.pyplot as plt

# 1. Дані та CSV

MONTHS = list(range(1, 25))
TEMPS  = [-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0,
          -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3]

def write_csv(fname, x, y):
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Month", "Temp"])
        w.writeheader()
        for xi, yi in zip(x, y):
            w.writerow({"Month": xi, "Temp": yi})

def read_csv(fname):
    x, y = [], []
    with open(fname, newline="") as f:
        for row in csv.DictReader(f):
            x.append(float(row["Month"]))
            y.append(float(row["Temp"]))
    return x, y

# 2. Формування матриці та вектора МНК

def form_matrix(x, m):
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum(x[k] ** (i + j) for k in range(len(x)))
    return A

def form_vector(x, y, m):
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * x[k] ** i for k in range(len(x)))
    return b

# 3. Метод Гауса з вибором головного елемента

def gauss_solve(A, b):
    n = len(b)
    # Копіюємо щоб не змінювати оригінал
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Прямий хід
    for k in range(n):
        # Вибір головного елемента по стовпцю
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[max_row] = M[max_row], M[k]

        for i in range(k + 1, n):
            if M[k][k] == 0:
                continue
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    # Зворотний хід
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        x_sol[i] = M[i][n]
        for j in range(i + 1, n):
            x_sol[i] -= M[i][j] * x_sol[j]
        x_sol[i] /= M[i][i]

    return x_sol

# 4. Обчислення многочлена

def polynomial(x, coef):
    result = []
    for xi in x:
        val = sum(coef[i] * xi ** i for i in range(len(coef)))
        result.append(val)
    return result

# 5. Дисперсія

def variance(y_true, y_approx):
    n = len(y_true)
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n

# 6. Головна

write_csv("temperature.csv", MONTHS, TEMPS)
x, y = read_csv("temperature.csv")

print("Табуляція вхідних даних:")
print(" Місяць | Температура")
for xi, yi in zip(x, y):
    print(f"   {int(xi):2d}   |   {yi:.1f}")

# Вибір оптимального степеня
print("\nДисперсія для різних степенів многочлена:")
variances = []
for m in range(1, 5):
    A   = form_matrix(x, m)
    b_v = form_vector(x, y, m)
    coef = gauss_solve(A, b_v)
    y_approx = polynomial(x, coef)
    var = variance(y, y_approx)
    variances.append(var)
    print(f"  m = {m}: дисперсія = {var:.6f}")

optimal_m = variances.index(min(variances)) + 1
print(f"\nОптимальний степінь: m = {optimal_m}")

# Апроксимація оптимальним многочленом
A    = form_matrix(x, optimal_m)
b_v  = form_vector(x, y, optimal_m)
coef = gauss_solve(A, b_v)

print("\nКоефіцієнти многочлена:")
for i, c in enumerate(coef):
    print(f"  a[{i}] = {c:.6f}")

y_approx = polynomial(x, coef)

# Похибка
error = [y[i] - y_approx[i] for i in range(len(x))]
print("\nПохибка апроксимації:")
print(" Місяць | Фактична | Апрокс. | Похибка")
for i in range(len(x)):
    print(f"   {int(x[i]):2d}   | {y[i]:8.2f} | {y_approx[i]:7.2f} | {error[i]:7.4f}")

# Прогноз на 3 місяці
x_future = [25.0, 26.0, 27.0]
y_future = polynomial(x_future, coef)
print("\nПрогноз на наступні 3 місяці:")
for xi, yi in zip(x_future, y_future):
    print(f"  Місяць {int(xi)}: {yi:.2f} °C")

# 7. Графік 1 — апроксимація

plt.figure(figsize=(10, 5))
plt.scatter(x, y, color="red", zorder=5, label="Фактичні дані")
plt.plot(x, y_approx, "b-", lw=2, label=f"Апроксимація (m={optimal_m})")
plt.scatter(x_future, y_future, color="green", zorder=5, marker="^", s=80, label="Прогноз")
plt.title("Апроксимація температури методом найменших квадратів")
plt.xlabel("Місяць")
plt.ylabel("Температура (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Графік 2 — похибка

plt.figure(figsize=(10, 5))
plt.plot(x, error, "r-", lw=2, label="Похибка")
plt.axhline(0, color="black", lw=1)
plt.title("Похибка апроксимації")
plt.xlabel("Місяць")
plt.ylabel("y - P(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Графік 3 — дисперсія від степеня

plt.figure(figsize=(7, 4))
plt.plot(range(1, 5), variances, "bo-", lw=2)
plt.scatter([optimal_m], [variances[optimal_m - 1]], color="red", zorder=5, s=100, label=f"Оптимум m={optimal_m}")
plt.title("Дисперсія від степеня многочлена")
plt.xlabel("Степінь m")
plt.ylabel("Дисперсія")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()