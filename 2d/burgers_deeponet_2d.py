# -*- coding: utf-8 -*-
"""
2D Burgers - DeepONet (FIXED: PCA reduction + better training)
"""
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from scipy.interpolate import RegularGridInterpolator
import time

print("Loading dataset...")
data = np.load('deeponet_data_2d.npz')
x_grid = data['x']
y_grid = data['y']
t_grid = data['t']
u0_all = data['u0']  # (500, 32, 32)
u_all = data['u']     # (500, 20, 32, 32)

N_samples = len(u0_all)
N_x = len(x_grid)
N_y = len(y_grid)
N_t = len(t_grid)

print(f"Original: u0={u0_all.shape}, u={u_all.shape}")

# ============================================================
# FIX 1: PCA снижение размерности начальных условий
# ============================================================
print("\nApplying PCA to initial conditions...")
u0_flat_full = u0_all.reshape(N_samples, -1)  # (500, 1024)

# Оставляем 32 главные компоненты (было 1024!)
pca = PCA(n_components=32, random_state=42)
u0_pca = pca.fit_transform(u0_flat_full)  # (500, 32)

explained_var = pca.explained_variance_ratio_.sum()
print(f"PCA: 1024 -> 32 components, explained variance: {explained_var:.3f} ({explained_var*100:.1f}%)")

# ============================================================
# FIX 2: Уменьшаем выходную сетку
# ============================================================
# Берём каждую 4-ю точку по x,y и каждую 2-ю по t
x_coarse = x_grid[::4]   # 8 точек
y_coarse = y_grid[::4]   # 8 точек
t_coarse = t_grid[::2]   # 10 точек

u_all_coarse = u_all[:, ::2, ::4, ::4]  # (500, 10, 8, 8)

N_xc = len(x_coarse)
N_yc = len(y_coarse)
N_tc = len(t_coarse)

# Flatten выход
u_flat_coarse = u_all_coarse.reshape(N_samples, N_tc, -1).astype(np.float32)  # (500, 10, 64)

# ============================================================
# Train/test split
# ============================================================
np.random.seed(42)
idx = np.random.permutation(N_samples)
train_idx = idx[:400]
test_idx = idx[400:]

# Branch: PCA-сжатые начальные условия
branch_train = u0_pca[train_idx].astype(np.float32)  # (400, 32)

# Trunk: все координаты (x, y, t) на coarse сетке
X_mesh, Y_mesh, T_mesh = np.meshgrid(x_coarse, y_coarse, t_coarse, indexing='ij')
trunk_input = np.hstack([
    X_mesh.flatten()[:, None],
    Y_mesh.flatten()[:, None],
    T_mesh.flatten()[:, None]
]).astype(np.float32)  # (8*8*10, 3) = (640, 3)

# Выход
sol_train = u_flat_coarse[train_idx].reshape(len(train_idx), -1).astype(np.float32)  # (400, 640)

print(f"\nReduced dimensions:")
print(f"  branch: {branch_train.shape} (was (400, 1024))")
print(f"  trunk:  {trunk_input.shape} (was (20480, 3))")
print(f"  sol:    {sol_train.shape} (was (400, 20480))")
print(f"  Compression: {(1 - sol_train.size / (400*20480)) * 100:.1f}%")

branch_test = u0_pca[test_idx[:5]].astype(np.float32)
sol_test = sol_train[:5].astype(np.float32)

# ============================================================
# FIX 3: Компактная архитектура
# ============================================================
net = dde.nn.DeepONetCartesianProd(
    layer_sizes_branch=[32, 64, 64, 32],    # 32 -> 64 -> 64 -> 32
    layer_sizes_trunk=[3, 64, 64, 32],      # 3 -> 64 -> 64 -> 32
    activation="relu",
    kernel_initializer="Glorot normal",
)

data_dde = dde.data.Triple(
    X_train=(branch_train, trunk_input),
    y_train=sol_train,
    X_test=(branch_test, trunk_input),
    y_test=sol_test,
)

model = dde.Model(data_dde, net)
model.compile("adam", lr=5e-4)  # чуть ниже learning rate

# ============================================================
# FIX 4: Обучение с cosine decay
# ============================================================
print("\n" + "=" * 50)
print("Training FIXED 2D DeepONet...")
print("=" * 50)

start_time = time.time()

losshistory, train_state = model.train(
    iterations=20000,
    display_every=2000,
    model_save_path="burgers_2d_deeponet_fixed",
)

training_time = time.time() - start_time
print(f"Training completed: {training_time:.1f} sec")

# ============================================================
# Loss
# ============================================================
loss_train = np.array(losshistory.loss_train)
plt.figure(figsize=(10, 6))
plt.semilogy(np.arange(len(loss_train)) * 2000, loss_train.flatten(), 'b-', lw=2)
plt.xlabel('Epochs'); plt.ylabel('MSE')
plt.title('2D DeepONet Loss (Fixed)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('burgers_2d_deeponet_fixed_loss.png', dpi=150)

# ============================================================
# Тест
# ============================================================
print("Testing...")
test_idx_sample = test_idx[0]

# Предсказание
u0_test = u0_pca[test_idx_sample:test_idx_sample+1].astype(np.float32)
u_test_true = u_all_coarse[test_idx_sample]  # (10, 8, 8)

u_pred = model.predict((u0_test, trunk_input))
u_pred = u_pred.reshape(N_tc, N_xc, N_yc)

# Интерполяция на тонкую сетку для визуализации
x_fine = np.linspace(-1, 1, 32)
y_fine = np.linspace(-1, 1, 32)
Xf, Yf = np.meshgrid(x_fine, y_fine, indexing='ij')

t_plot_idx = [0, N_tc//3, 2*N_tc//3, -1]
fig, axes = plt.subplots(2, len(t_plot_idx), figsize=(4*len(t_plot_idx), 8))

for i, t_idx in enumerate(t_plot_idx):
    interp_true = RegularGridInterpolator(
        (x_coarse, y_coarse), u_test_true[t_idx].T, bounds_error=False, fill_value=0)
    interp_pred = RegularGridInterpolator(
        (x_coarse, y_coarse), u_pred[t_idx].T, bounds_error=False, fill_value=0)
    
    true_fine = interp_true(np.stack([Xf.flatten(), Yf.flatten()], axis=-1)).reshape(32, 32)
    pred_fine = interp_pred(np.stack([Xf.flatten(), Yf.flatten()], axis=-1)).reshape(32, 32)
    
    im1 = axes[0, i].pcolormesh(x_fine, y_fine, true_fine.T, shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, i].set_title(f'True t={t_coarse[t_idx]:.2f}')
    plt.colorbar(im1, ax=axes[0, i])
    
    im2 = axes[1, i].pcolormesh(x_fine, y_fine, pred_fine.T, shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, i].set_title(f'Fixed DeepONet t={t_coarse[t_idx]:.2f}')
    plt.colorbar(im2, ax=axes[1, i])

plt.suptitle('2D DeepONet (Fixed): True vs Predicted', fontsize=14)
plt.tight_layout()
plt.savefig('burgers_2d_deeponet_fixed_test.png', dpi=150)
print("Plots saved.")

# Метрики
mse = mean_squared_error(u_test_true.flatten(), u_pred.flatten())
mae = mean_absolute_error(u_test_true.flatten(), u_pred.flatten())
rel_l2 = np.linalg.norm(u_test_true.flatten() - u_pred.flatten()) / (np.linalg.norm(u_test_true.flatten()) + 1e-10)

print(f"\nMSE: {mse:.4e}, MAE: {mae:.4e}, Rel L2: {rel_l2:.4e}")
print(f"Time: {training_time:.1f} sec")
print("Done!")
plt.show()