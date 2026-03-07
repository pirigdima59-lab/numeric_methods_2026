import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. Запит до API

coords_str = (
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

url = f"https://api.open-elevation.com/api/v1/lookup?locations={coords_str}"

try:
    response = requests.get(url, timeout=15)
    results = response.json()["results"]
except Exception as e:
    print("Помилка при запиті до API:", e)
    exit()

# 2. Табуляція вузлів

n = len(results)
print("Кількість вузлів:", n)
print("\nТабуляція вузлів:")
print(" № | Latitude   | Longitude  | Elevation (m)")
for i, p in enumerate(results):
    print(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")

# 3. Кумулятивна відстань

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

coords     = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances  = [0.0]
for i in range(1, n):
    distances.append(distances[-1] + haversine(*coords[i-1], *coords[i]))

print("\nТабуляція (відстань, висота):")
print(" № | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:12.2f} | {elevations[i]:8.2f}")

# 4. Метод прогонки

def thomas(alpha, beta, gamma, delta):
    n = len(beta)
    A = np.zeros(n)
    B = np.zeros(n)

    # Пряма прогонка
    for i in range(1, n - 1):
        denom = alpha[i] * A[i-1] + beta[i]
        A[i]  = -gamma[i] / denom
        B[i]  = (delta[i] - alpha[i] * B[i-1]) / denom

    # Зворотна прогонка
    c = np.zeros(n)
    for i in range(n - 2, 0, -1):
        c[i] = A[i] * c[i+1] + B[i]

    return c

# 5. Побудова кубічних сплайнів

def build_cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)

    # Формування трьохдіагональної матриці
    alpha = np.zeros(n)
    beta  = np.ones(n)
    gamma = np.zeros(n)
    delta = np.zeros(n)

    for i in range(1, n - 1):
        alpha[i] = h[i-1]
        beta[i]  = 2 * (h[i-1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    c = thomas(alpha, beta, gamma, delta)

    a = y[:-1]
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(n - 1):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    return a, b, c[:-1], d

# 6. Обчислення значення сплайна

def evaluate_spline(x_eval, x, a, b, c, d):
    y_eval = np.zeros_like(x_eval)
    for idx, xe in enumerate(x_eval):
        for i in range(len(x) - 1):
            if x[i] <= xe <= x[i+1] or (i == len(x) - 2 and xe > x[i+1]):
                dx = xe - x[i]
                y_eval[idx] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
                break
    return y_eval

# 7. Коефіцієнти сплайнів

x_data = np.array(distances)
y_data = np.array(elevations)

a, b, c, d = build_cubic_spline(x_data, y_data)

print("\nКоефіцієнти кубічних сплайнів:")
print("  i |        a_i |        b_i |        c_i |        d_i")
for i in range(len(a)):
    print(f"{i:3d} | {a[i]:10.2f} | {b[i]:10.4f} | {c[i]:10.6f} | {d[i]:10.8f}")

# 8. Похибка при різній кількості вузлів

x_dense = np.linspace(x_data[0], x_data[-1], 500)
y_full  = evaluate_spline(x_dense, x_data, a, b, c, d)

for n_pts in [10, 15, 20]:
    idx  = np.round(np.linspace(0, len(x_data) - 1, n_pts)).astype(int)
    xs, ys = x_data[idx], y_data[idx]
    as_, bs, cs, ds = build_cubic_spline(xs, ys)
    y_sub = evaluate_spline(x_dense, xs, as_, bs, cs, ds)
    err   = np.abs(y_full - y_sub)
    print(f"\n===== {n_pts} вузлів =====")
    print(f"Максимальна похибка: {np.max(err)}")
    print(f"Середня похибка: {np.mean(err)}")

# 9. Графік 1 — вплив кількості вузлів

plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_full, label="21 вузол (еталон)", linewidth=2)
for n_pts, col in [(10, "orange"), (15, "green"), (20, "red")]:
    idx  = np.round(np.linspace(0, len(x_data) - 1, n_pts)).astype(int)
    xs, ys = x_data[idx], y_data[idx]
    as_, bs, cs, ds = build_cubic_spline(xs, ys)
    plt.plot(x_dense, evaluate_spline(x_dense, xs, as_, bs, cs, ds),
             color=col, label=f"{n_pts} вузлів")
plt.title("Вплив кількості вузлів")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Графік 2 — похибка апроксимації

plt.figure(figsize=(10, 6))
for n_pts, col in [(10, "blue"), (15, "orange"), (20, "green")]:
    idx  = np.round(np.linspace(0, len(x_data) - 1, n_pts)).astype(int)
    xs, ys = x_data[idx], y_data[idx]
    as_, bs, cs, ds = build_cubic_spline(xs, ys)
    err = np.abs(y_full - evaluate_spline(x_dense, xs, as_, bs, cs, ds))
    plt.plot(x_dense, err, color=col, label=f"{n_pts} вузлів")
plt.title("Похибка апроксимації")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("|похибка| (м)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Характеристики маршруту

total_distance = distances[-1]
total_ascent   = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
total_descent  = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n))

print(f"\nЗагальна довжина маршруту (м): {total_distance:.2f}")
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"Сумарний спуск (м): {total_descent:.2f}")

# 12. Аналіз градієнта

grad = np.gradient(y_full, x_dense) * 100
print(f"Максимальний підйом (%): {np.max(grad):.2f}")
print(f"Максимальний спуск (%): {np.min(grad):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad)):.2f}")

# 13. Механічна енергія

mass   = 80
energy = mass * 9.81 * total_ascent
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")
print(f"Енергія (ккал): {energy / 4184:.2f}")