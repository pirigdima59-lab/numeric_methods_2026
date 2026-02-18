import requests
import numpy as np
import matplotlib.pyplot as plt


print("--- Крок 1-3: Отримання та табуляція даних ---")
coords_str = "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
url = f"https://api.open-elevation.com/api/v1/lookup?locations={coords_str}"

try:
    response = requests.get(url)
    data = response.json()
    results = data["results"]
except Exception as e:
    print("Помилка при запиті до API:", e)
    exit()

n_points = len(results)
print("Кількість вузлів:", n_points)
print("\nТабуляція вузлів:")
print(" ID | Latitude  | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")


print("\n--- Крок 4: Обчислення кумулятивної відстані ---")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0.0]

for i in range(1, n_points):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

print("\nТабуляція (відстань, висота):")
print(" ID | Distance (m) | Elevation (m)")
for i in range(n_points):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")


def build_cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)

    # Ініціалізація коефіцієнтів системи рівнянь
    alpha = np.zeros(n)
    beta = np.ones(n)
    gamma = np.zeros(n)
    delta = np.zeros(n)

    # Формування трьохдіагональної матриці
    for i in range(1, n - 1):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Метод прогонки (Thomas algorithm)
    A = np.zeros(n)
    B = np.zeros(n)

    # Пряма прогонка
    for i in range(1, n - 1):
        denominator = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denominator
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denominator

    # Зворотна прогонка для знаходження коефіцієнтів C
    c = np.zeros(n)
    for i in range(n - 2, 0, -1):
        c[i] = A[i] * c[i + 1] + B[i]

    # Знаходження коефіцієнтів a, b, d
    a = y[:-1]
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c[:-1], d, c


def evaluate_spline(x_eval, x, a, b, c, d):
    y_eval = np.zeros_like(x_eval)
    for idx, xe in enumerate(x_eval):
        # Знаходимо потрібний інтервал
        for i in range(len(x) - 1):
            if x[i] <= xe <= x[i + 1] or (i == len(x) - 2 and xe >= x[i + 1]):
                dx = xe - x[i]
                y_eval[idx] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
                break
    return y_eval


x_data = np.array(distances)
y_data = np.array(elevations)

a, b, c_spline, d, c_full = build_cubic_spline(x_data, y_data)

print("\n--- Крок 8-9: Коефіцієнти кубічних сплайнів ---")
print("  i |        a_i |        b_i |        c_i |        d_i")
for i in range(len(a)):
    print(f"{i:3d} | {a[i]:10.2f} | {b[i]:10.4f} | {c_spline[i]:10.6f} | {d[i]:10.8f}")


print("\n--- Крок 10: Побудова графіків для різної кількості вузлів ---")
x_dense = np.linspace(x_data[0], x_data[-1], 500)

plt.figure(figsize=(12, 8))

# Базовий графік з усіма точками
y_dense_all = evaluate_spline(x_dense, x_data, a, b, c_spline, d)
plt.plot(x_dense, y_dense_all, label='Сплайн (усі вузли)', color='black', linewidth=2)
plt.scatter(x_data, y_data, color='red', label='Вузли API', zorder=5)

# Дослідження впливу кількості вузлів (10, 15, 20)
for nodes_count in [10, 15, 20]:
    if nodes_count > len(x_data):
        continue
    # Вибираємо рівномірно розподілені індекси
    idx = np.round(np.linspace(0, len(x_data) - 1, nodes_count)).astype(int)
    x_sub = x_data[idx]
    y_sub = y_data[idx]

    a_sub, b_sub, c_sub, d_sub, _ = build_cubic_spline(x_sub, y_sub)
    y_dense_sub = evaluate_spline(x_dense, x_sub, a_sub, b_sub, c_sub, d_sub)

    plt.plot(x_dense, y_dense_sub, linestyle='--', label=f'Сплайн ({nodes_count} вузлів)')

plt.title("Профіль висоти: Заросляк - Говерла (Кубічні сплайни)")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.show()


# Додаткові завдання

print("\n--- Додаткове завдання ---")
# 1. Характеристики маршруту
total_distance = distances[-1]
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n_points))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n_points))

print(f"Загальна довжина маршруту (м): {total_distance:.2f}")
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"Сумарний спуск (м): {total_descent:.2f}")

# 2. Аналіз градієнта
grad_full = np.gradient(y_dense_all, x_dense) * 100
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

# 3. Механічна енергія підйому
mass = 80  # кг
g = 9.81
energy = mass * g * total_ascent
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")
print(f"Енергія (ккал): {energy / 4184:.2f}")