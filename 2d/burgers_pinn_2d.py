# -*- coding: utf-8 -*-
"""
2D Burgers Equation - PINN
u_t + u*u_x + u*u_y = nu*(u_xx + u_yy)
"""
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf
import time

# ============================================================
# Параметры
# ============================================================
nu = 0.01 / np.pi

# Геометрия: (x,y) in [-1,1]x[-1,1], t in [0,1]
geom = dde.geometry.Rectangle([-1, -1], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

print(f"nu = {nu:.6f}")

# ============================================================
# Загрузка эталона (для визуализации)
# ============================================================
ref = np.load('reference_2d.npz')
x_ref = ref['x']
y_ref = ref['y']
t_ref = ref['t']
u0_ref = ref['u0']
u_ref = ref['u']

# ============================================================
# PDE (2D Burgers)
# ============================================================
def pde(x, u):
    """
    x: (x, y, t)
    u: solution
    Returns residual: u_t + u*u_x + u*u_y - nu*(u_xx + u_yy)
    """
    # Первые производные
    du_x = dde.grad.jacobian(u, x, i=0, j=0)  # du/dx
    du_y = dde.grad.jacobian(u, x, i=0, j=1)  # du/dy
    du_t = dde.grad.jacobian(u, x, i=0, j=2)  # du/dt
    
    # Вторые производные
    du_xx = dde.grad.hessian(u, x, i=0, j=0)  # d2u/dx2
    du_yy = dde.grad.hessian(u, x, i=1, j=1)  # d2u/dy2
    
    # Невязка
    return du_t + u * du_x + u * du_y - nu * (du_xx + du_yy)

# ============================================================
# Начальные и граничные условия
# ============================================================
# IC: u(x, y, 0) из эталона
def ic_func(x):
    """Интерполируем эталонное начальное условие"""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (x_ref, y_ref), u0_ref.T, bounds_error=False, fill_value=0.0
    )
    return interp(x[:, :2]).reshape(-1, 1)

ic = dde.icbc.IC(geomtime, ic_func, lambda _, on: on)

# BC: периодические границы (упрощение — Dirichlet 0)
bc = dde.icbc.DirichletBC(
    geomtime, 
    lambda x: np.zeros((len(x), 1)), 
    lambda _, on: on
)

# ============================================================
# Данные и сеть
# ============================================================
data = dde.data.TimePDE(
    geomtime, pde, [ic, bc],
    num_domain=1500,    # меньше точек для скорости
    num_boundary=300,
    num_initial=300,
)

net = dde.nn.FNN(
    layer_sizes=[3] + [40] * 5 + [1],  # 3 входа (x,y,t)
    activation="tanh",
    kernel_initializer="Glorot normal",
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# ============================================================
# Обучение
# ============================================================
print("\n" + "=" * 50)
print("Training 2D PINN...")
print("=" * 50)

start_time = time.time()

losshistory, train_state = model.train(
    iterations=15000,
    display_every=1500,
    model_save_path="burgers_2d_pinn",
)

training_time = time.time() - start_time
print(f"Training completed: {training_time:.2f} sec")

# ============================================================
# Loss график
# ============================================================
loss_train = np.array(losshistory.loss_train)
plt.figure(figsize=(10, 6))
plt.semilogy(np.arange(len(loss_train)) * 1500, loss_train.sum(axis=1), 'k-', lw=2, label='Total')
plt.semilogy(np.arange(len(loss_train)) * 1500, loss_train[:, 0], 'r--', lw=1, label='PDE')
plt.semilogy(np.arange(len(loss_train)) * 1500, loss_train[:, 1], 'g--', lw=1, label='IC')
plt.semilogy(np.arange(len(loss_train)) * 1500, loss_train[:, 2], 'b--', lw=1, label='BC')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.title('2D PINN Training Loss')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('burgers_2d_pinn_loss.png', dpi=150)
print("Loss plot saved.")

# ============================================================
# Визуализация
# ============================================================
print("\nGenerating predictions...")

# Сетка для предсказания
N_plot = 64
x_plot = np.linspace(-1, 1, N_plot)
y_plot = np.linspace(-1, 1, N_plot)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot, indexing='ij')

# Предсказываем в 4 момента времени
t_slices = [0.0, 0.33, 0.66, 1.0]
u_pred_slices = []
u_true_slices = []

for t_val in t_slices:
    # Предсказание PINN
    xyt = np.hstack([
        X_plot.flatten()[:, None],
        Y_plot.flatten()[:, None],
        np.full((N_plot*N_plot, 1), t_val)
    ])
    u_pred = model.predict(xyt).reshape(N_plot, N_plot)
    u_pred_slices.append(u_pred)
    
    # Эталон (интерполяция)
    t_idx = np.argmin(np.abs(t_ref - t_val))
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (x_ref, y_ref), u_ref[t_idx].T, bounds_error=False, fill_value=0.0
    )
    xy = np.hstack([X_plot.flatten()[:, None], Y_plot.flatten()[:, None]])
    u_true = interp(xy).reshape(N_plot, N_plot)
    u_true_slices.append(u_true)

# Рисуем
fig, axes = plt.subplots(3, 4, figsize=(18, 12))

for i, t_val in enumerate(t_slices):
    # Эталон
    im1 = axes[0, i].pcolormesh(x_plot, y_plot, u_true_slices[i].T, 
                                 shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, i].set_title(f'Exact: t={t_val}')
    axes[0, i].set_xlabel('x'); axes[0, i].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, i])
    
    # PINN
    im2 = axes[1, i].pcolormesh(x_plot, y_plot, u_pred_slices[i].T,
                                 shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, i].set_title(f'PINN: t={t_val}')
    axes[1, i].set_xlabel('x'); axes[1, i].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, i])
    
    # Ошибка
    error = np.abs(u_pred_slices[i] - u_true_slices[i])
    im3 = axes[2, i].pcolormesh(x_plot, y_plot, error.T,
                                 shading='auto', cmap='hot', vmin=0, vmax=0.5)
    axes[2, i].set_title(f'Error: t={t_val}')
    axes[2, i].set_xlabel('x'); axes[2, i].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2, i])

plt.suptitle('2D Burgers: PINN vs Exact', fontsize=16)
plt.tight_layout()
plt.savefig('burgers_2d_pinn_heatmaps.png', dpi=150)
print("Heatmaps saved.")

# ============================================================
# Метрики
# ============================================================
all_pred = np.array([u.flatten() for u in u_pred_slices]).flatten()
all_true = np.array([u.flatten() for u in u_true_slices]).flatten()

from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(all_true, all_pred)
mae = mean_absolute_error(all_true, all_pred)
rel_l2 = np.linalg.norm(all_true - all_pred) / np.linalg.norm(all_true)

print("\n" + "=" * 50)
print("2D PINN ACCURACY")
print("=" * 50)
print(f"MSE:              {mse:.6e}")
print(f"MAE:              {mae:.6e}")
print(f"Relative L2:      {rel_l2:.6e}")
print(f"Training time:    {training_time:.2f} sec")
print("=" * 50)

plt.show()
print("Done!")