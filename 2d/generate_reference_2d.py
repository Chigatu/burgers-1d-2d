# -*- coding: utf-8 -*-
"""Generate 2D Burgers reference & DeepONet dataset"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time

nu = 0.01 / np.pi

# Сетки
N_x = 64
N_y = 64
N_t = 40
N_samples = 500  # для DeepONet

x_grid = np.linspace(-1, 1, N_x)
y_grid = np.linspace(-1, 1, N_y)
t_grid = np.linspace(0, 1, N_t)
dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]

def solve_2d_burgers(u0, t_grid, nu):
    """Split-step Fourier for 2D Burgers (stable)"""
    Nx, Ny = u0.shape
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    u = u0.copy()
    u_snapshots = np.zeros((len(t_grid), Nx, Ny))
    dt = t_grid[1] - t_grid[0]
    
    for step, t_val in enumerate(t_grid):
        u_snapshots[step] = u.copy()
        
        # Split-step
        # 1) Half-step diffusion
        u_hat = np.fft.fft2(u)
        u_hat = u_hat * np.exp(-nu * K2 * dt * 0.5)
        u = np.fft.ifft2(u_hat).real
        
        # 2) Nonlinear (Lax-Friedrichs for stability)
        ux = np.gradient(u, dx, axis=0)
        uy = np.gradient(u, dy, axis=1)
        u = u - dt * u * (ux + uy)
        u = np.clip(u, -5, 5)
        
        # 3) Half-step diffusion
        u_hat = np.fft.fft2(u)
        u_hat = u_hat * np.exp(-nu * K2 * dt * 0.5)
        u = np.fft.ifft2(u_hat).real
    
    return u_snapshots

# ============================================================
# 1. Эталонное решение (один случай — для PINN)
# ============================================================
print("Generating reference solution for 2D PINN...")

# Начальное условие: 3 вихря
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
u0_ref = (
    1.0 * np.exp(-((X-0.3)**2 + (Y-0.3)**2) / 0.15**2)
    - 0.8 * np.exp(-((X+0.3)**2 + (Y+0.3)**2) / 0.2**2)
    + 0.5 * np.exp(-((X-0.0)**2 + (Y+0.5)**2) / 0.12**2)
)

print(f"u0_ref range: [{u0_ref.min():.3f}, {u0_ref.max():.3f}]")

u_ref = solve_2d_burgers(u0_ref, t_grid, nu)
print(f"u_ref shape: {u_ref.shape}")
print(f"u_ref range: [{u_ref.min():.3f}, {u_ref.max():.3f}]")
print(f"NaN count: {np.isnan(u_ref).sum()}")

np.savez('reference_2d.npz', x=x_grid, y=y_grid, t=t_grid, u0=u0_ref, u=u_ref)
print("Reference saved: reference_2d.npz")

# Визуализация эталона
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for i, t_idx in enumerate([0, 10, 25, -1]):
    im = axes[i].pcolormesh(x_grid, y_grid, u_ref[t_idx].T, shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[i].set_title(f't = {t_grid[t_idx]:.2f}')
    axes[i].set_xlabel('x'); axes[i].set_ylabel('y')
    plt.colorbar(im, ax=axes[i])
plt.suptitle('2D Burgers Reference Solution', fontsize=14)
plt.tight_layout()
plt.savefig('burgers_2d_reference.png', dpi=150)
print("Reference plot saved.")

# ============================================================
# 2. Датасет для DeepONet (500 случайных вихревых полей)
# ============================================================
print(f"\nGenerating {N_samples} samples for DeepONet 2D...")
np.random.seed(42)

# Грубая сетка для скорости
N_x_coarse = 32
N_y_coarse = 32
N_t_coarse = 20

x_coarse = np.linspace(-1, 1, N_x_coarse)
y_coarse = np.linspace(-1, 1, N_y_coarse)
t_coarse = np.linspace(0, 1, N_t_coarse)

Xc, Yc = np.meshgrid(x_coarse, y_coarse, indexing='ij')

u0_all = np.zeros((N_samples, N_x_coarse, N_y_coarse), dtype=np.float32)
u_all = np.zeros((N_samples, N_t_coarse, N_x_coarse, N_y_coarse), dtype=np.float32)

for i in range(N_samples):
    n_vortices = np.random.randint(2, 6)
    u0 = np.zeros((N_x_coarse, N_y_coarse))
    
    for _ in range(n_vortices):
        a = np.random.uniform(-1.0, 1.0)
        cx = np.random.uniform(-0.6, 0.6)
        cy = np.random.uniform(-0.6, 0.6)
        w = np.random.uniform(0.12, 0.35)
        u0 += a * np.exp(-((Xc - cx)**2 + (Yc - cy)**2) / w**2)
    
    u0_all[i] = u0.astype(np.float32)
    
    # Решаем на coarse сетке
    u_snap = solve_2d_burgers(u0, t_coarse, nu)
    u_all[i] = u_snap.astype(np.float32)
    
    if (i+1) % 100 == 0:
        print(f"  Generated {i+1}/{N_samples} samples")

print(f"NaN check: {np.isnan(u_all).sum()}, Inf: {np.isinf(u_all).sum()}")

np.savez('deeponet_data_2d.npz',
         x=x_coarse, y=y_coarse, t=t_coarse,
         u0=u0_all, u=u_all)
print("DeepONet 2D dataset saved: deeponet_data_2d.npz")

# Превью датасета
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for i in range(3):
    idx = np.random.randint(0, N_samples)
    im0 = axes[0, i].pcolormesh(x_coarse, y_coarse, u0_all[idx].T, shading='auto', cmap='RdBu_r')
    axes[0, i].set_title(f'Sample {idx}: u(x,y,0)')
    plt.colorbar(im0, ax=axes[0, i])
    
    im1 = axes[1, i].pcolormesh(x_coarse, y_coarse, u_all[idx, -1].T, shading='auto', cmap='RdBu_r')
    axes[1, i].set_title(f'Sample {idx}: u(x,y,1.0)')
    plt.colorbar(im1, ax=axes[1, i])

plt.suptitle('2D DeepONet Dataset Preview', fontsize=14)
plt.tight_layout()
plt.savefig('deeponet_2d_dataset_preview.png', dpi=150)
print("Dataset preview saved.")

plt.show()
print("\nDone! All 2D data generated.")