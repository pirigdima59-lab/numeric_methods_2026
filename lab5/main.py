"""
Лабораторна робота №5. Формула Сімпсона, Рунге-Ромберг, Ейткен, адаптивний алгоритм.
f(x) = 50 + 20*sin(pi*x/12) + 5*e^(-0.2*(x-12)^2), x in [0, 24]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import integrate
import os


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)


a, b = 0, 24

# Точне значення інтегралу
I0, _ = integrate.quad(f, a, b)

print("ЛАБОРАТОРНА РОБОТА №5")
print("f(x) = 50 + 20sin(px/12) + 5e^(-0.2(x-12)^2), x in [0,24]")

print("\n2. ТОЧНЕ ЗНАЧЕННЯ ІНТЕГРАЛУ")
print(f"   I0 = {I0:.8f}")


def simpson(func, a, b, N):
    """Складова формула Сімпсона, N парне."""
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)
    s = y[0] + y[-1] + 4 * np.sum(y[1:N:2]) + 2 * np.sum(y[2:N-1:2])
    return s * h / 3


print("\n3. ФОРМУЛА СІМПСОНА (перевірка N=10)")
print(f"   I(10) = {simpson(f, a, b, 10):.8f}")

# Залежність похибки від N
print("\n4. ЗАЛЕЖНІСТЬ ПОХИБКИ ВІД N")

N_range = range(10, 1002, 2)
errors_N, vals_N = [], []
for N in N_range:
    val = simpson(f, a, b, N)
    vals_N.append(val)
    errors_N.append(abs(val - I0))

TARGET_EPS = 1e-12
N_opt, eps_opt = None, None
for i, N in enumerate(N_range):
    if errors_N[i] <= TARGET_EPS:
        N_opt, eps_opt = N, errors_N[i]
        break

if N_opt is None:
    idx_min = np.argmin(errors_N)
    N_opt = list(N_range)[idx_min]
    eps_opt = errors_N[idx_min]

print(f"   Цільова точність: 1e-12")
print(f"   N_opt = {N_opt},  досягнута точність = {eps_opt:.2e}")

# N0 кратне 8
print("\n5. ПОХИБКА ПРИ N0")
N0_raw = max(16, round(N_opt / 10))
N0 = N0_raw + (8 - N0_raw % 8) % 8
if N0 < 16:
    N0 = 16
I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)
print(f"   N0 = {N0}  (приблизно N_opt/10, кратне 8)")
print(f"   I(N0) = {I_N0:.8f}")
print(f"   eps0 = {eps0:.2e}")

# Рунге-Ромберг
print("\n6. МЕТОД РУНГЕ-РОМБЕРГА")
N0_double = N0 * 2
I_N0d = simpson(f, a, b, N0_double)
I_R = I_N0d + (I_N0d - I_N0) / (2**4 - 1)
eps_R = abs(I_R - I0)
print(f"   I(N0)    = {I_N0:.10f}   N0 = {N0}")
print(f"   I(2*N0)  = {I_N0d:.10f}   2*N0 = {N0_double}")
print(f"   I_R = I(2*N0) + (I(2*N0) - I(N0)) / 15")
print(f"   I_R = {I_R:.10f}")
print(f"   epsR = {eps_R:.2e}")
print(f"   Покращення у {eps0/eps_R:.1f} разів")

# Ейткен
print("\n7. МЕТОД ЕЙТКЕНА")
N1, N2, N3 = N0, N0 * 2, N0 * 4
I1, I2, I3 = simpson(f, a, b, N1), simpson(f, a, b, N2), simpson(f, a, b, N3)

num = (I2 - I1)**2
den = I3 - 2 * I2 + I1
I_A = I1 - num / den if abs(den) > 1e-20 else I3
eps_A = abs(I_A - I0)

if abs(I3 - I2) > 1e-20 and abs(I2 - I1) > 1e-20:
    p_est = np.log(abs(I2 - I1) / abs(I3 - I2)) / np.log(2)
else:
    p_est = float('nan')

print(f"   I1 = {I1:.10f}   N1 = {N1}")
print(f"   I2 = {I2:.10f}   N2 = {N2}")
print(f"   I3 = {I3:.10f}   N3 = {N3}")
print(f"   I* = {I_A:.10f}")
print(f"   epsA = {eps_A:.2e}")
print(f"   Оцінений порядок p ≈ {p_est:.2f}")

# Адаптивний алгоритм
print("\n9. АДАПТИВНИЙ АЛГОРИТМ")

eval_count = [0]


def adaptive_simpson(func, a, b, tol, depth=0, max_depth=50):
    """Рекурсивний адаптивний Сімпсон."""
    mid = (a + b) / 2
    fa, fm, fb = func(a), func(mid), func(b)
    eval_count[0] += 3
    S = (b - a) / 6 * (fa + 4 * fm + fb)

    m1, m2 = (a + mid) / 2, (mid + b) / 2
    fm1, fm2 = func(m1), func(m2)
    eval_count[0] += 2
    S2 = ((mid - a) / 6 * (fa + 4 * fm1 + fm) +
          (b - mid) / 6 * (fm + 4 * fm2 + fb))

    if depth >= max_depth or abs(S2 - S) < 15 * tol:
        return S2 + (S2 - S) / 15
    return (adaptive_simpson(func, a, mid, tol / 2, depth + 1, max_depth) +
            adaptive_simpson(func, mid, b, tol / 2, depth + 1, max_depth))


tol_values = [1e-3, 1e-5, 1e-7, 1e-9]
adaptive_results = []

print(f"\n   {'eps':>10}   {'Інтеграл':>14}   {'Похибка':>10}   {'Обчислень':>10}")
for tol in tol_values:
    eval_count[0] = 0
    I_ad = adaptive_simpson(f, a, b, tol)
    eps_ad = abs(I_ad - I0)
    adaptive_results.append((tol, I_ad, eps_ad, eval_count[0]))
    print(f"   {tol:>10.0e}   {I_ad:>14.8f}   {eps_ad:>10.2e}   {eval_count[0]:>10}")

# Підсумкова таблиця
print("\nПІДСУМКОВА ТАБЛИЦЯ")
print(f"\n   {'Метод':<28}   {'Значення':>14}   {'Похибка':>10}")
rows = [
    ("Точне значення",               I0,                    None),
    (f"Сімпсон N0={N0}",             I_N0,                  eps0),
    ("Рунге-Ромберг",                I_R,                   eps_R),
    ("Ейткен",                       I_A,                   eps_A),
    ("Адаптивний (eps=1e-7)",        adaptive_results[2][1], adaptive_results[2][2]),
]
for name, val, err in rows:
    err_str = f"{err:.2e}" if err is not None else "точне"
    print(f"   {name:<28}   {val:>14.8f}   {err_str:>10}")


# Графіки
COLOR_BG    = '#0f172a'
COLOR_PANEL = '#1e293b'
COLOR_ACC1  = '#38bdf8'
COLOR_ACC2  = '#f472b6'
COLOR_ACC3  = '#a3e635'
COLOR_ACC4  = '#fb923c'
COLOR_ACC5  = '#c084fc'
COLOR_TEXT  = '#e2e8f0'
COLOR_MUTED = '#94a3b8'


def style_ax(ax, title):
    ax.set_facecolor(COLOR_PANEL)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.set_title(title, color=COLOR_TEXT, fontsize=11, fontweight='bold', pad=10)
    ax.xaxis.label.set_color(COLOR_MUTED)
    ax.yaxis.label.set_color(COLOR_MUTED)
    ax.grid(True, color='#1e3a5f', alpha=0.5, linewidth=0.6)


fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor(COLOR_BG)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Графік функції
ax1 = fig.add_subplot(gs[0, :2])
x_arr = np.linspace(a, b, 500)
ax1.plot(x_arr, f(x_arr), color=COLOR_ACC1, linewidth=2.2, label='f(x)')
ax1.fill_between(x_arr, f(x_arr), alpha=0.12, color=COLOR_ACC1)
ax1.set_xlabel('x (год)')
ax1.set_ylabel('Навантаження f(x)')
ax1.legend(framealpha=0, labelcolor=COLOR_TEXT, fontsize=9)
style_ax(ax1, 'f(x) = 50 + 20sin(px/12) + 5e^(-0.2(x-12)^2)')

# Похибка від N
ax2 = fig.add_subplot(gs[0, 2])
N_list = list(N_range)
valid = [(n, e) for n, e in zip(N_list, errors_N) if e > 0]
nv, ev = zip(*valid)
ax2.loglog(nv, ev, color=COLOR_ACC3, linewidth=2, label='|I(N) - I0|')
ref_n = np.array(nv[:30], dtype=float)
ax2.loglog(ref_n, 1e4 * ref_n**(-4.0), '--', color=COLOR_MUTED,
           linewidth=1, alpha=0.7, label='O(N^-4)')
ax2.axvline(N0, color=COLOR_ACC2, linestyle='--', linewidth=1.3, label=f'N0={N0}')
ax2.set_xlabel('N')
ax2.set_ylabel('Похибка')
ax2.legend(framealpha=0, labelcolor=COLOR_TEXT, fontsize=8)
style_ax(ax2, 'Похибка від кількості вузлів N')

# Порівняння методів
ax3 = fig.add_subplot(gs[1, 0])
methods = ['Сімпсон\nN0', 'Рунге-\nРомберг', 'Ейткен']
errs    = [eps0, eps_R, eps_A]
colors  = [COLOR_ACC1, COLOR_ACC2, COLOR_ACC3]
bars = ax3.bar(methods, errs, color=colors, edgecolor='#334155', linewidth=0.8, width=0.5)
ax3.set_yscale('log')
ax3.set_ylabel('Похибка (log)')
for bar, e in zip(bars, errs):
    ax3.text(bar.get_x() + bar.get_width() / 2, e * 2,
             f'{e:.1e}', ha='center', va='bottom', fontsize=8, color=COLOR_TEXT)
style_ax(ax3, 'Порівняння точності методів')

# Адаптивний: похибка vs eps
ax4 = fig.add_subplot(gs[1, 1])
tols_ad  = [r[0] for r in adaptive_results]
eps_ad_l = [r[2] for r in adaptive_results]
ax4.loglog(tols_ad, eps_ad_l, 'o-', color=COLOR_ACC4, linewidth=2, markersize=7)
ax4.loglog(tols_ad, tols_ad, '--', color=COLOR_MUTED, linewidth=1, alpha=0.7, label='y = eps')
ax4.set_xlabel('Задана точність eps')
ax4.set_ylabel('Реальна похибка')
ax4.legend(framealpha=0, labelcolor=COLOR_TEXT, fontsize=9)
style_ax(ax4, 'Адаптивний: похибка vs eps')

# Адаптивний: кількість обчислень vs eps
ax5 = fig.add_subplot(gs[1, 2])
evals_ad = [r[3] for r in adaptive_results]
ax5.semilogx(tols_ad, evals_ad, 's-', color=COLOR_ACC5, linewidth=2, markersize=7)
ax5.set_xlabel('Задана точність eps')
ax5.set_ylabel('Кількість обчислень f(x)')
for x_pt, y_pt in zip(tols_ad, evals_ad):
    ax5.text(x_pt, y_pt + max(evals_ad) * 0.02, str(y_pt),
             ha='center', fontsize=8, color=COLOR_TEXT)
style_ax(ax5, 'Адаптивний: обчислення vs eps')

fig.suptitle('Числове інтегрування: формула Сімпсона та методи підвищення точності',
             color=COLOR_TEXT, fontsize=12, fontweight='bold', y=0.99)

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab5_simpson_plots.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLOR_BG)
plt.close()
print(f"\nГрафіки збережено: {save_path}")