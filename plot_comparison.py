# -*- coding: utf-8 -*-
"""
Comparative Loss Plot: PINN vs PINN+RAR vs DeepONet (1D & 2D)
FINAL VERSION with all 5 models
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Цветовая схема
colors = {
    'PINN 1D': '#2196F3',
    'RAR 1D': '#FF9800',
    'DeepONet 1D': '#4CAF50',
    'PINN 2D': '#9C27B0',
    'DeepONet 2D': '#F44336',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================
# График 1: 1D сравнение
# ============================================================
ax = axes[0]

# PINN 1D: шаг 500, 10000 эпох
epochs_pinn_1d = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
                           5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])
loss_pinn_1d = np.array([4.55e+00, 4.31e-02, 3.24e-02, 2.00e-02, 1.01e-02, 4.48e-03,
                         3.10e-03, 6.09e-03, 1.33e-02, 1.91e-03, 1.38e-03, 1.17e-03,
                         1.06e-03, 9.44e-04, 9.50e-04, 1.18e-03, 2.32e-03, 7.60e-04,
                         7.10e-04, 9.11e-04, 3.73e-04])

ax.semilogy(epochs_pinn_1d, loss_pinn_1d, '-', color=colors['PINN 1D'], linewidth=2,
            marker='o', markersize=4, markevery=2, label='PINN 1D (L2=8.27)')

# RAR 1D: шаг 500, 10000 эпох (этап дообучения)
epochs_rar_1d = np.arange(0, 10500, 500)
loss_rar_1d = np.array([5.41e+00, 4.58e-02, 3.94e-02, 3.89e-02, 3.49e-02, 2.39e-02,
                        1.47e-02, 7.70e-03, 5.61e-03, 4.68e-03, 3.85e-03, 2.91e-03,
                        1.13e-03, 7.35e-04, 5.59e-04, 4.98e-04, 4.01e-04, 3.53e-04,
                        3.09e-04, 2.78e-04, 2.54e-04])

ax.semilogy(epochs_rar_1d, loss_rar_1d, '--', color=colors['RAR 1D'], linewidth=2,
            marker='s', markersize=4, markevery=2, label='PINN+RAR 1D (L2=8.37)')

# DeepONet 1D: шаг 2000, 20000 эпох
epochs_don_1d = np.arange(0, 21000, 2000)
loss_don_1d = np.array([2.87e-01, 4.26e-02, 3.89e-02, 3.63e-02, 3.44e-02, 3.33e-02,
                        3.23e-02, 3.13e-02, 3.06e-02, 2.99e-02, 2.94e-02])

ax.semilogy(epochs_don_1d, loss_don_1d, '-.', color=colors['DeepONet 1D'], linewidth=2,
            marker='^', markersize=4, markevery=1, label='DeepONet 1D (L2=0.62) ★')

ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Loss (MSE, log scale)', fontsize=12)
ax.set_title('1D Burgers: Training Loss Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim([-200, 10200])

# ============================================================
# График 2: 2D сравнение (исправленный)
# ============================================================
ax = axes[1]

# PINN 2D: шаг 1500, 15000 эпох
epochs_pinn_2d = np.arange(0, 16500, 1500)
loss_pinn_2d = np.array([2.23e-02, 1.12e-03, 9.04e-04, 6.19e-04, 3.35e-04, 1.93e-04,
                         1.01e-04, 5.83e-05, 4.37e-05, 4.30e-05, 1.89e-05])

ax.semilogy(epochs_pinn_2d, loss_pinn_2d, '-', color=colors['PINN 2D'], linewidth=2,
            marker='o', markersize=6, label='PINN 2D (L2=0.53)')

# DeepONet 2D MINI: шаг 2000, 20000 эпох
epochs_don_2d = np.arange(0, 21000, 2000)
loss_don_2d = np.array([5.48e-02, 1.30e-02, 1.04e-02, 8.31e-03, 7.07e-03, 6.06e-03,
                        5.69e-03, 5.19e-03, 4.89e-03, 4.32e-03, 4.14e-03])

# Плавное падение — от 0.055 до 0.004
ax.semilogy(epochs_don_2d, loss_don_2d, '-.', color=colors['DeepONet 2D'], linewidth=2.5,
            marker='^', markersize=7, label='DeepONet 2D MINI (L2=0.28) ★')

ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Loss (MSE, log scale)', fontsize=12)
ax.set_title('2D Burgers: Training Loss Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim([-500, 20500])  # симметрично с данными

# Аннотация на графике
ax.annotate('Loss ↓ 13× in 20k epochs',
            xy=(10000, 1e-2), fontsize=9, color='darkred',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.suptitle('Physics-Informed vs Operator Networks: Convergence Comparison\n'
             '1D (left): PINN vs RAR vs DeepONet  |  2D (right): PINN vs DeepONet MINI',
             fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('comparison_loss_all.png', dpi=200, bbox_inches='tight')
print("Saved: comparison_loss_all.png")

# ============================================================
# Столбчатая диаграмма: метрики
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

methods = ['PINN\n1D', 'PINN+RAR\n1D', 'DeepONet\n1D', 'PINN\n2D', 'DeepONet\n2D MINI']
rel_l2 = [8.27, 8.37, 0.62, 0.53, 0.28]
times = [117, 166, 310, 202, 69]
bar_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, rel_l2, width, label='Relative L2 Error',
               color=bar_colors, edgecolor='black', linewidth=0.8)

ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, times, width, label='Training Time (sec)',
                color='lightgray', edgecolor='black', linewidth=0.8, alpha=0.7)

ax.set_ylabel('Relative L2 Error', fontsize=13, color='darkred')
ax2.set_ylabel('Training Time (seconds)', fontsize=13, color='gray')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_title('Model Comparison: Accuracy vs Training Time', fontsize=15, fontweight='bold')

# Значения над столбцами
for bar, val in zip(bars1, rel_l2):
    y_pos = bar.get_height() + 0.4
    color = 'darkgreen' if val < 1 else 'darkred'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

for bar, val in zip(bars2, times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val}s', ha='center', va='bottom', fontsize=9, color='gray')

# ★ рядом с лучшим
ax.text(4 - width/2, rel_l2[4] + 0.5, '★ BEST', ha='center', fontsize=12,
        color='darkgreen', fontweight='bold')

# Легенда
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# Пояснительная подпись
ax.text(2, max(rel_l2) * 0.9, '* L2 < 1 — хорошее качество\n'
        '* PINN не требует данных\n'
        '* DeepONet 2D MINI — лучший результат',
        fontsize=9, color='gray', fontstyle='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('comparison_metrics_bar.png', dpi=200, bbox_inches='tight')
print("Saved: comparison_metrics_bar.png")

# ============================================================
# Сводная таблица в консоль
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)
print(f"{'Model':<20} {'Rel L2':<10} {'MSE':<12} {'Time':<10} {'Data?':<8}")
print("-" * 70)
print(f"{'PINN 1D':<20} {8.27:<10.2f} {3.70e-01:<12.2e} {'117s':<10} {'No':<8}")
print(f"{'PINN+RAR 1D':<20} {8.37:<10.2f} {3.79e-01:<12.2e} {'166s':<10} {'No':<8}")
print(f"{'DeepONet 1D':<20} {0.62:<10.2f} {1.06e-01:<12.2e} {'310s':<10} {'Yes':<8}")
print(f"{'PINN 2D':<20} {0.53:<10.2f} {4.67e-03:<12.2e} {'202s':<10} {'No':<8}")
print(f"{'DeepONet 2D MINI':<20} {0.28:<10.2f} {'—':<12} {'69s':<10} {'Yes':<8}")
print("-" * 70)
print("★ BEST: DeepONet 2D MINI (L2=0.28, 69 seconds)")
print("=" * 70)

plt.show()
print("\nDone! All comparison plots updated.")