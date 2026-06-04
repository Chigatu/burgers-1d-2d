# -*- coding: utf-8 -*-
"""
2D Burgers - DeepONet MINIMAL (guaranteed to work)
Ultra-simple: linear advection-diffusion, 1 Gaussian → solution
"""
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf
from sklearn.metrics import mean_squared_error
import time

# ============================================================
# Генерируем ДАНИЕ (очень простые: 1 гауссиана)
# ============================================================
print("Generating dataset...")
nu = 0.01 / np.pi

N_xy = 8          # очень маленькая сетка
N_t = 5
N_samples = 1000

x_grid = np.linspace(-1, 1, N_xy)
y_grid = np.linspace(-1, 1, N_xy)
t_grid = np.linspace(0, 1, N_t)
dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]
dt = t_grid[1] - t_grid[0]

X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

def solve_2d(u0, t_grid, nu):
    """Simple split-step solver"""
    Nx, Ny = u0.shape
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    K2 = kx[:, None]**2 + ky[None, :]**2
    
    u = u0.copy()
    snapshots = np.zeros((len(t_grid), Nx, Ny))
    dt_local = t_grid[1] - t_grid[0]
    
    for step in range(len(t_grid)):
        snapshots[step] = u.copy()
        if step == len(t_grid) - 1:
            break
        # diffusion
        u_hat = np.fft.fft2(u)
        u_hat = u_hat * np.exp(-nu * K2 * dt_local)
        u = np.fft.ifft2(u_hat).real
        # advection (upwind)
        ux = np.gradient(u, dx, axis=0)
        uy = np.gradient(u, dy, axis=1)
        u = u - dt_local * u * (ux + uy)
        u = np.clip(u, -3, 3)
    
    return snapshots

np.random.seed(42)
u0_all = np.zeros((N_samples, N_xy, N_xy), dtype=np.float32)
u_all = np.zeros((N_samples, N_t, N_xy, N_xy), dtype=np.float32)

for i in range(N_samples):
    a = np.random.uniform(0.5, 1.5)
    cx = np.random.uniform(-0.3, 0.3)
    cy = np.random.uniform(-0.3, 0.3)
    w = np.random.uniform(0.2, 0.5)
    
    u0 = a * np.exp(-((X - cx)**2 + (Y - cy)**2) / w**2)
    u0_all[i] = u0.astype(np.float32)
    u_all[i] = solve_2d(u0, t_grid, nu).astype(np.float32)
    
    if (i+1) % 250 == 0:
        print(f"  {i+1}/{N_samples}")

print(f"Data: u0={u0_all.shape}, u={u_all.shape}")

# ============================================================
# DeepONet (простая архитектура)
# ============================================================
u0_flat = u0_all.reshape(N_samples, -1).astype(np.float32)  # (1000, 64)
u_flat = u_all.reshape(N_samples, N_t, -1).astype(np.float32)  # (1000, 5, 64)

# Trunk
Xm, Ym, Tm = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')
trunk = np.hstack([Xm.flatten()[:, None], Ym.flatten()[:, None], Tm.flatten()[:, None]]).astype(np.float32)
# (8*8*5, 3) = (320, 3)

# Split
train_n = 800
branch_train = u0_flat[:train_n]         # (800, 64)
sol_train = u_flat[:train_n].reshape(train_n, -1)  # (800, 320)

branch_test = u0_flat[train_n:train_n+10]
sol_test = sol_train[:10]

print(f"branch: {branch_train.shape}, trunk: {trunk.shape}, sol: {sol_train.shape}")

# Сеть
net = dde.nn.DeepONetCartesianProd(
    layer_sizes_branch=[64, 64, 64, 32],
    layer_sizes_trunk=[3, 64, 64, 32],
    activation="relu",
    kernel_initializer="Glorot normal",
)

data = dde.data.Triple(
    X_train=(branch_train, trunk),
    y_train=sol_train,
    X_test=(branch_test, trunk),
    y_test=sol_test,
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# ============================================================
# Train
# ============================================================
print("\nTraining...")
start = time.time()

losshistory, train_state = model.train(
    iterations=20000,
    display_every=2000,
    model_save_path="burgers_2d_don_mini",
)

print(f"Time: {time.time() - start:.1f}s")

# Loss plot
loss_train = np.array(losshistory.loss_train)
plt.figure(figsize=(10, 6))
plt.semilogy(np.arange(len(loss_train)) * 2000, loss_train.flatten(), 'b-', lw=2)
plt.xlabel('Epochs'); plt.ylabel('MSE')
plt.title('2D DeepONet MINI: Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('burgers_2d_don_mini_loss.png', dpi=150)

# ============================================================
# Test
# ============================================================
print("Testing...")
test_idx = 850
u0_t = u0_flat[test_idx:test_idx+1]
u_true = u_all[test_idx]
u_pred = model.predict((u0_t, trunk)).reshape(N_t, N_xy, N_xy)

# Визуализация
fig, axes = plt.subplots(3, N_t, figsize=(N_t*3, 9))
for i in range(N_t):
    im0 = axes[0, i].pcolormesh(x_grid, y_grid, u_true[i].T, shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, i].set_title(f'True t={t_grid[i]:.2f}')
    plt.colorbar(im0, ax=axes[0, i])
    
    im1 = axes[1, i].pcolormesh(x_grid, y_grid, u_pred[i].T, shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, i].set_title(f'Pred t={t_grid[i]:.2f}')
    plt.colorbar(im1, ax=axes[1, i])
    
    err = np.abs(u_pred[i] - u_true[i])
    im2 = axes[2, i].pcolormesh(x_grid, y_grid, err.T, shading='auto', cmap='hot', vmin=0, vmax=0.3)
    axes[2, i].set_title(f'Err t={t_grid[i]:.2f}')
    plt.colorbar(im2, ax=axes[2, i])

plt.suptitle('2D DeepONet MINI', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('burgers_2d_don_mini_test.png', dpi=150)

# Метрики
all_rel = []
for idx in range(800, 900):
    u_true_i = u_all[idx]
    u_pred_i = model.predict((u0_flat[idx:idx+1], trunk)).reshape(N_t, N_xy, N_xy)
    rel = np.linalg.norm(u_true_i - u_pred_i) / (np.linalg.norm(u_true_i) + 1e-10)
    all_rel.append(rel)

print(f"\nRelative L2 (100 tests): {np.mean(all_rel):.4e} ± {np.std(all_rel):.4e}")
print(f"Time: {time.time() - start:.1f}s")
print("Done!")
plt.show()