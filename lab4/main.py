
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. Функція та її аналітична похідна

def M(t):
    """Вологість ґрунту"""
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_exact_derivative(t):
    """Аналітична перша похідна M'(t) = -5*e^(-0.1t) + 5*cos(t)"""
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t0 = 1.0
exact = M_exact_derivative(t0)


print("  ЛАБОРАТОРНА РОБОТА №5")
print("  Чисельне диференціювання функції M(t) = 50e^(-0.1t) + 5sin(t)")

# Крок 1: Аналітичне розв'язання

print("\n 1. АНАЛІТИЧНЕ РОЗВ'ЯЗАННЯ")
print(f"   M'(t) = -5·e^(-0.1t) + 5·cos(t)")
print(f"   M'(1) = -5·e^(-0.1) + 5·cos(1)")
print(f"         = -5·{np.exp(-0.1):.6f} + 5·{np.cos(1):.6f}")
print(f"         = {exact:.6f}")


# Крок 2: Центральна різниця — залежність похибки від h

print("\n 2. ЗАЛЕЖНІСТЬ ПОХИБКИ ВІД КРОКУ h (центральна різниця) ")

def central_diff(f, t, h):
    """Центральна різниця: (f(t+h) - f(t-h)) / (2h)"""
    return (f(t + h) - f(t - h)) / (2 * h)

steps = [10**(-k) for k in range(1, 9)]
errors = []
approx_vals = []

print(f"\n   {'h':>12}  {'D(h)':>12}  {'Похибка':>12}")
for h in steps:
    d = central_diff(M, t0, h)
    err = abs(d - exact)
    approx_vals.append(d)
    errors.append(err)
    print(f"   {h:>12.1e}  {d:>12.7f}  {err:>12.2e}")

# Знаходимо оптимальний крок
opt_idx = np.argmin(errors)
h_opt = steps[opt_idx]
print(f"\n   Оптимальний крок: h_opt = {h_opt:.0e}")
print(f"   Найкраща точність: {errors[opt_idx]:.2e}")


# Крок 3–5: Два кроки сітки

print("\n 3–5. ДВА КРОКИ СІТКИ ")
h1 = 0.01
h2 = h1 / 2  # h/2

D_h1 = central_diff(M, t0, h1)
D_h2 = central_diff(M, t0, h2)
err_h1 = abs(D_h1 - exact)
err_h2 = abs(D_h2 - exact)

print(f"   h  = {h1}  →  D(h)  = {D_h1:.7f}  похибка = {err_h1:.2e}")
print(f"   h/2 = {h2} →  D(h/2)= {D_h2:.7f}  похибка = {err_h2:.2e}")


# Крок 6: Метод Рунге-Ромберга

print("\n 6. МЕТОД РУНГЕ-РОМБЕРГА ")
# Для центральної різниці порядок p = 2
# R = D(h/2) + (D(h/2) - D(h)) / (2^p - 1)
p = 2
R = D_h2 + (D_h2 - D_h1) / (2**p - 1)
err_R = abs(R - exact)

print(f"   D(h)  = {D_h1:.7f}")
print(f"   D(h/2)= {D_h2:.7f}")
print(f"   R = D(h/2) + (D(h/2) - D(h)) / ({2**p} - 1)")
print(f"   R = {D_h2:.7f} + ({D_h2:.7f} - {D_h1:.7f}) / {2**p - 1}")
print(f"   R = {R:.7f}")
print(f"   Похибка Рунге-Ромберга: {err_R:.2e}")
print(f"   Покращення у {err_h1/err_R:.1f} разів (порівняно з h={h1})")


# Крок 7: Метод Ейткена

print("\n 7. МЕТОД ЕЙТКЕНА ")
h3 = h1 / 4  # третій крок
D_h3 = central_diff(M, t0, h3)
err_h3 = abs(D_h3 - exact)

print(f"   D1 = D(h)   = {D_h1:.7f}   (h={h1})")
print(f"   D2 = D(h/2) = {D_h2:.7f}   (h/2={h2})")
print(f"   D3 = D(h/4) = {D_h3:.7f}   (h/4={h3})")

# Ейткен: A* = D1 - (D2-D1)^2 / (D3 - 2*D2 + D1)
numerator   = (D_h2 - D_h1)**2
denominator = D_h3 - 2*D_h2 + D_h1

if abs(denominator) > 1e-15:
    A_star = D_h1 - numerator / denominator
else:
    A_star = D_h3  # fallback

err_A = abs(A_star - exact)

# Оцінка порядку точності
if abs(D_h3 - D_h2) > 1e-15 and abs(D_h2 - D_h1) > 1e-15:
    p_est = np.log(abs(D_h2 - D_h1) / abs(D_h3 - D_h2)) / np.log(2)
else:
    p_est = float('nan')

print(f"\n   A* = D1 - (D2-D1)² / (D3 - 2·D2 + D1)")
print(f"   A* = {A_star:.7f}")
print(f"   Точне значення: {exact:.7f}")
print(f"   Похибка методу Ейткена: {err_A:.2e}")
print(f"   Оцінений порядок точності p ≈ {p_est:.2f}")


# Підсумкова таблиця

print("\n ПІДСУМКОВА ТАБЛИЦЯ ")
print(f"\n   {'Метод':<30}  {'Значення':>12}  {'Похибка':>12}")
print(f"   {'Аналітичне':30}  {exact:12.7f}  {'—':>12}")
print(f"   {'Центральна різниця (h=0.01)':30}  {D_h1:12.7f}  {err_h1:12.2e}")
print(f"   {'Центральна різниця (h/2)':30}  {D_h2:12.7f}  {err_h2:12.2e}")
print(f"   {'Центральна різниця (h/4)':30}  {D_h3:12.7f}  {err_h3:12.2e}")
print(f"   {'Рунге-Ромберг':30}  {R:12.7f}  {err_R:12.2e}")
print(f"   {'Ейткен':30}  {A_star:12.7f}  {err_A:12.2e}")


# Графіки

fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor('#0f172a')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

COLOR_BG    = '#0f172a'
COLOR_PANEL = '#1e293b'
COLOR_ACC1  = '#38bdf8'
COLOR_ACC2  = '#f472b6'
COLOR_ACC3  = '#a3e635'
COLOR_ACC4  = '#fb923c'
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

# ── Графік 1: M(t) ──
ax1 = fig.add_subplot(gs[0, 0])
t_arr = np.linspace(0, 20, 400)
ax1.plot(t_arr, M(t_arr), color=COLOR_ACC1, linewidth=2.2, label='M(t)')
ax1.axvline(t0, color=COLOR_ACC2, linestyle='--', linewidth=1.4, label=f't₀={t0}')
ax1.scatter([t0], [M(t0)], color=COLOR_ACC2, s=60, zorder=5)
ax1.set_xlabel('t')
ax1.set_ylabel('M(t)')
ax1.legend(framealpha=0, labelcolor=COLOR_TEXT, fontsize=9)
style_ax(ax1, 'Вологість ґрунту M(t)')

# ── Графік 2: M'(t) ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t_arr, M_exact_derivative(t_arr), color=COLOR_ACC3, linewidth=2.2, label="M'(t) аналітично")
ax2.axvline(t0, color=COLOR_ACC2, linestyle='--', linewidth=1.4)
ax2.axhline(exact, color=COLOR_ACC4, linestyle=':', linewidth=1.2, label=f"M'(1)={exact:.4f}")
ax2.scatter([t0], [exact], color=COLOR_ACC2, s=60, zorder=5)
ax2.set_xlabel('t')
ax2.set_ylabel("M'(t)")
ax2.legend(framealpha=0, labelcolor=COLOR_TEXT, fontsize=9)
style_ax(ax2, "Похідна M'(t)")

# ── Графік 3: похибка від h ──
ax3 = fig.add_subplot(gs[1, 0])
valid = [(h, e) for h, e in zip(steps, errors) if e > 0]
hv, ev = zip(*valid)
ax3.loglog(hv, ev, 'o-', color=COLOR_ACC1, linewidth=2, markersize=6, label='Похибка |D(h) - M\'(1)|')
ax3.axvline(h_opt, color=COLOR_ACC2, linestyle='--', linewidth=1.4, label=f'h_opt={h_opt:.0e}')
ref_h = np.array(hv[:5])
ax3.loglog(ref_h, ref_h**2 * 0.1, '--', color=COLOR_MUTED, linewidth=1, alpha=0.6, label='O(h²)')
ax3.set_xlabel('Крок h')
ax3.set_ylabel('Похибка')
ax3.legend(framealpha=0, labelcolor=COLOR_TEXT, fontsize=8)
style_ax(ax3, 'Залежність похибки від кроку h')

# ── Графік 4: порівняння методів ──
ax4 = fig.add_subplot(gs[1, 1])
methods = ['CD\nh=0.01', 'CD\nh/2', 'CD\nh/4', 'Рунге-\nРомберг', 'Ейткен']
errs    = [err_h1, err_h2, err_h3, err_R, err_A]
colors  = [COLOR_ACC1, COLOR_ACC1, COLOR_ACC1, COLOR_ACC2, COLOR_ACC3]
bars = ax4.bar(methods, errs, color=colors, edgecolor='#334155', linewidth=0.8, width=0.6)
ax4.set_yscale('log')
ax4.set_ylabel('Похибка (log)')
for bar, e in zip(bars, errs):
    ax4.text(bar.get_x() + bar.get_width()/2, e * 1.5, f'{e:.1e}',
             ha='center', va='bottom', fontsize=8, color=COLOR_TEXT)
style_ax(ax4, 'Порівняння точності методів')

fig.suptitle('Чисельне диференціювання: M(t) = 50e⁻⁰·¹ᵗ + 5sin(t)',
             color=COLOR_TEXT, fontsize=13, fontweight='bold', y=0.98)

import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab4_plots.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLOR_BG)
plt.close()
print(f"\nГрафіки збережено: {save_path}")
print("Готово!")