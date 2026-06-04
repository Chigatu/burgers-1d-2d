# -*- coding: utf-8 -*-
"""Generate stable dataset for DeepONet using simple initial conditions"""
import numpy as np
import matplotlib.pyplot as plt
import os

nu = 0.01 / np.pi
N_x = 128
N_t = 50
N_samples = 1000

def solve_burgers_analytical(x, t, a, c, w):
    """
    Solve Burgers for Gaussian initial condition: u(x,0) = a * exp(-(x-c)^2/w^2)
    Using split-step Fourier (stable version with clipping)
    """
    N = len(x)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    
    # Начальное условие
    u = a * np.exp(-(x - c)**2 / w**2)
    u_snapshots = np.zeros((len(t), N))
    
    dt = t[1] - t[0]
    
    for step, t_val in enumerate(t):
        u_snapshots[step] = u.copy()
        
        # Полушаг: линейная часть
        u_hat = np.fft.fft(u)
        u_hat = u_hat * np.exp(-nu * k**2 * dt * 0.5)
        u = np.fft.ifft(u_hat).real
        
        # Нелинейная часть
        u = u - dt * u * np.gradient(u, dx)
        
        # Клиппинг для стабильности
        u = np.clip(u, -5, 5)
        
        # Полушаг: линейная часть
        u_hat = np.fft.fft(u)
        u_hat = u_hat * np.exp(-nu * k**2 * dt * 0.5)
        u = np.fft.ifft(u_hat).real
    
    return u_snapshots

print("Generating stable DeepONet dataset...")

x_grid = np.linspace(-1, 1, N_x)
t_grid = np.linspace(0, 1, N_t)

np.random.seed(42)
u0_all = np.zeros((N_samples, N_x))
u_all = np.zeros((N_samples, N_t, N_x))

params_list = []

for i in range(N_samples):
    # Параметры гауссиан с ограничениями для стабильности
    a = np.random.uniform(-1.5, 1.5)
    c = np.random.uniform(-0.6, 0.6)
    w = np.random.uniform(0.15, 0.5)
    params_list.append((a, c, w))
    
    u0 = a * np.exp(-(x_grid - c)**2 / w**2)
    u0_all[i] = u0
    
    u_snap = solve_burgers_analytical(x_grid, t_grid, a, c, w)
    u_all[i] = u_snap
    
    if (i+1) % 200 == 0:
        print(f"  Generated {i+1}/{N_samples} samples")

# Проверка на NaN
nan_count = np.isnan(u_all).sum()
inf_count = np.isinf(u_all).sum()
print(f"\nNaN count: {nan_count}, Inf count: {inf_count}")
print(f"u range: min={u_all.min():.4f}, max={u_all.max():.4f}")

if nan_count > 0 or inf_count > 0:
    print("WARNING: Data contains NaN/Inf! Cleaning...")
    u_all = np.nan_to_num(u_all, nan=0.0, posinf=1.0, neginf=-1.0)

# Сохраняем
save_path = 'deeponet_data_1d.npz'
np.savez(save_path, x=x_grid, t=t_grid, u0=u0_all, u=u_all)
print(f"Dataset saved to {save_path}")

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i in range(3):
    idx = np.random.randint(0, N_samples)
    axes[0, i].plot(x_grid, u0_all[idx], 'b-', linewidth=2)
    axes[0, i].set_title(f'Sample {idx}: u(x,0)')
    axes[0, i].set_xlabel('x'); axes[0, i].set_ylabel('u')
    axes[0, i].grid(True, alpha=0.3)
    
    im = axes[1, i].pcolormesh(x_grid, t_grid, u_all[idx], shading='auto', cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    axes[1, i].set_title(f'Sample {idx}: u(x,t)')
    axes[1, i].set_xlabel('x'); axes[1, i].set_ylabel('t')
    plt.colorbar(im, ax=axes[1, i])

plt.suptitle('DeepONet Dataset: 1D Burgers Equation', fontsize=14)
plt.tight_layout()
plt.savefig('deeponet_dataset_preview.png', dpi=150)
print("Preview saved: deeponet_dataset_preview.png")
plt.show()