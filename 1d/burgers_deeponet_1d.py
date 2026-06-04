# -*- coding: utf-8 -*-
"""
1D Burgers - DeepONet (Deep Operator Network)
Learns the operator G: u(x,0) -> u(x,t)
"""
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf
from scipy.interpolate import RegularGridInterpolator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

print("Loading dataset...")

# Загружаем датасет
data = np.load('deeponet_data_1d.npz')
x_grid = data['x']
t_grid = data['t']
u0_all = data['u0']
u_all = data['u']

print(f"x_grid: {x_grid.shape}")
print(f"t_grid: {t_grid.shape}")
print(f"u0_all: {u0_all.shape}")
print(f"u_all: {u_all.shape}")

N_samples = u0_all.shape[0]
N_sensors = 128

# Train/test split
np.random.seed(42)
idx = np.random.permutation(N_samples)
train_idx = idx[:800]
test_idx = idx[800:]

# Более грубая сетка для trunk (быстрее обучение)
N_x_coarse = 64
N_t_coarse = 32

x_coarse = np.linspace(-1, 1, N_x_coarse)
t_coarse = np.linspace(0, 1, N_t_coarse)

X_mesh_coarse, T_mesh_coarse = np.meshgrid(x_coarse, t_coarse)
trunk_train = np.hstack([
    X_mesh_coarse.flatten()[:, None],
    T_mesh_coarse.flatten()[:, None]
]).astype(np.float32)

# Интерполируем обучающие решения на новую сетку
print("Interpolating training data...")
u_train_coarse = np.zeros((len(train_idx), N_t_coarse, N_x_coarse), dtype=np.float32)
for i, idx_val in enumerate(train_idx):
    interp = RegularGridInterpolator((t_grid, x_grid), u_all[idx_val])
    u_train_coarse[i] = interp(
        np.stack([T_mesh_coarse.flatten(), X_mesh_coarse.flatten()], axis=-1)
    ).reshape(N_t_coarse, N_x_coarse)

sol_train = u_train_coarse.reshape(len(train_idx), -1).astype(np.float32)
branch_train = u0_all[train_idx].astype(np.float32)

print(f"branch_train: {branch_train.shape}")
print(f"trunk_train: {trunk_train.shape}")
print(f"sol_train: {sol_train.shape}")

# Тестовые данные
u0_test_sample = u0_all[test_idx[:5]].astype(np.float32)
sol_test_sample = sol_train[:5].astype(np.float32)

print(f"Data types: branch={branch_train.dtype}, trunk={trunk_train.dtype}, sol={sol_train.dtype}")

# ============================================================
# DeepONet архитектура (DeepXDE 1.15+)
# ============================================================
net = dde.nn.DeepONetCartesianProd(
    layer_sizes_branch=[N_sensors, 128, 128, 100],
    layer_sizes_trunk=[2, 128, 128, 100],
    activation="relu",
    kernel_initializer="Glorot normal",
)

print(f"Branch net: {[N_sensors, 128, 128, 100]}")
print(f"Trunk net: {[2, 128, 128, 100]}")

# Данные
data = dde.data.Triple(
    X_train=(branch_train, trunk_train),
    y_train=sol_train,
    X_test=(u0_test_sample, trunk_train),
    y_test=sol_test_sample,
)

model = dde.Model(data, net)

# ============================================================
# Обучение (full-batch, без batch_size)
# ============================================================
model.compile("adam", lr=1e-3)
print("\n" + "=" * 50)
print("Training DeepONet...")
print("=" * 50)

start_time = time.time()

losshistory, train_state = model.train(
    iterations=20000,
    display_every=2000,
    model_save_path="burgers_1d_deeponet",
)

training_time = time.time() - start_time
print(f"\nTraining completed! Time: {training_time:.2f} sec")

# ============================================================
# Loss график
# ============================================================
loss_train = np.array(losshistory.loss_train)
plt.figure(figsize=(10, 6))
plt.semilogy(np.arange(len(loss_train)) * 2000, loss_train.flatten(), 'b-', linewidth=2)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('DeepONet Training Loss', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('burgers_1d_deeponet_loss.png', dpi=150)
print("Loss plot saved.")

# ============================================================
# Тестирование: предсказание для нового начального условия
# ============================================================
print("\nTesting DeepONet on unseen initial condition...")

test_idx_sample = test_idx[0]
u0_test = u0_all[test_idx_sample:test_idx_sample+1].astype(np.float32)
u_test_true = u_all[test_idx_sample]

# Предсказание на полной сетке для красивого графика
X_fine, T_fine = np.meshgrid(x_grid, t_grid)
trunk_test = np.hstack([
    X_fine.flatten()[:, None],
    T_fine.flatten()[:, None]
]).astype(np.float32)

u_pred = model.predict((u0_test, trunk_test))
u_pred = u_pred.reshape(len(t_grid), len(x_grid))

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Начальное условие
axes[0].plot(x_grid, u0_test[0], 'b-', linewidth=2)
axes[0].set_title('Input: Initial condition u(x,0)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u')
axes[0].grid(True, alpha=0.3)

# Истинное решение
im1 = axes[1].pcolormesh(x_grid, t_grid, u_test_true, shading='auto', cmap='RdBu_r', vmin=-2, vmax=2)
axes[1].set_title('True solution u(x,t)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
plt.colorbar(im1, ax=axes[1])

# Предсказание DeepONet
im2 = axes[2].pcolormesh(x_grid, t_grid, u_pred, shading='auto', cmap='RdBu_r', vmin=-2, vmax=2)
axes[2].set_title('DeepONet prediction')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')
plt.colorbar(im2, ax=axes[2])

plt.suptitle('DeepONet: 1D Burgers Equation', fontsize=15)
plt.tight_layout()
plt.savefig('burgers_1d_deeponet_test.png', dpi=150)
print("Test prediction saved.")

# Карта ошибки
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

error = np.abs(u_pred - u_test_true)
im3 = axes[0].pcolormesh(x_grid, t_grid, error, shading='auto', cmap='hot')
axes[0].set_title('Absolute Error |pred - true|')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
plt.colorbar(im3, ax=axes[0])

# Срез при t=1.0
axes[1].plot(x_grid, u_test_true[-1], 'b-', linewidth=2, label='True')
axes[1].plot(x_grid, u_pred[-1], 'r--', linewidth=2, label='DeepONet')
axes[1].set_title('Slice at t = 1.0')
axes[1].set_xlabel('x')
axes[1].set_ylabel('u')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('burgers_1d_deeponet_error.png', dpi=150)
print("Error plots saved.")

# ============================================================
# Метрики
# ============================================================
mse = mean_squared_error(u_test_true.flatten(), u_pred.flatten())
mae = mean_absolute_error(u_test_true.flatten(), u_pred.flatten())
rel_l2 = np.linalg.norm(u_test_true - u_pred) / np.linalg.norm(u_test_true)

print("\n" + "=" * 50)
print("DEEPONET ACCURACY METRICS")
print("=" * 50)
print(f"MSE:              {mse:.6e}")
print(f"MAE:              {mae:.6e}")
print(f"Relative L2:      {rel_l2:.6e}")
print(f"Training time:    {training_time:.2f} sec")
print("=" * 50)

print("\nDone! All DeepONet plots saved.")
plt.show()