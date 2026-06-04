# -*- coding: utf-8 -*-
"""
1D Burgers Equation - PINN (Windows-compatible)
u_t + u * u_x - nu * u_xx = 0,  x in [-1, 1], t in [0, 1]
Initial: u(x,0) = -sin(pi * x)
Boundary: u(-1,t) = u(1,t) = 0
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Отключаем лишние сообщения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # для Windows: интерактивные окна
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf
import time

print("Libraries loaded successfully!")
print(f"DeepXDE version: {dde.__version__}")
print(f"TensorFlow version: {tf.__version__}")

# ============================================================
# БЛОК 1: Параметры задачи
# ============================================================
nu = 0.01 / np.pi  # вязкость

# Геометрия: x в [-1, 1], t в [0, 1]
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

print(f"Viscosity nu = {nu:.6f}")

# ============================================================
# БЛОК 2: Аналитическое решение (Коул-Хопф) для проверки
# ============================================================
def exact_solution(x, t):
    """
    Load precomputed reference solution.
    Works with both 1D and 2D inputs.
    """
    import os
    
    # Загружаем один раз и кешируем
    if not hasattr(exact_solution, 'cache'):
        ref_path = os.path.join(os.path.dirname(__file__), 'reference_1d.npz')
        data = np.load(ref_path)
        exact_solution.cache = {
            'x': data['x'],
            't': data['t'],
            'u': data['u']
        }
    
    cache = exact_solution.cache
    x_ref = cache['x']
    t_ref = cache['t']
    u_ref = cache['u']
    
    # Приводим к двумерным массивам (time, space)
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)
    
    if x.ndim == 1 and t.ndim == 1:
        # Два 1D массива — создаём сетку
        X, T = np.meshgrid(x, t)
    else:
        # Уже двумерные
        X = x
        T = t
    
    result = np.zeros_like(X)
    
    # Для каждой точки находим ближайший индекс
    for i in range(T.shape[0]):
        t_val = T[i, 0]  # время одинаково для всей строки
        t_idx = np.argmin(np.abs(t_ref - t_val))
        
        for j in range(X.shape[1]):
            x_val = X[i, j]
            x_idx = np.argmin(np.abs(x_ref - x_val))
            result[i, j] = u_ref[t_idx, x_idx]
    
    return result.squeeze()

print("Exact solution function defined.")

# ============================================================
# БЛОК 3: Определение PDE (уравнение Бюргерса)
# ============================================================
def pde(x, u):
    """
    Residual: u_t + u*u_x - nu*u_xx
    """
    du_x = dde.grad.jacobian(u, x, i=0, j=0)   # du/dx
    du_t = dde.grad.jacobian(u, x, i=0, j=1)   # du/dt
    du_xx = dde.grad.hessian(u, x, i=0, j=0)   # d2u/dx2
    
    residual = du_t + u * du_x - nu * du_xx
    return residual

print("PDE residual defined.")

# ============================================================
# БЛОК 4: Начальные и граничные условия
# ============================================================
def ic_func(x):
    """Initial condition: u(x,0) = -sin(pi*x)"""
    return -np.sin(np.pi * x[:, 0:1])

def on_initial(x, on_initial):
    return on_initial

def bc_func(x):
    """Boundary condition: u(-1,t) = u(1,t) = 0"""
    return np.zeros((len(x), 1))

def on_boundary(x, on_boundary):
    return on_boundary

ic = dde.icbc.IC(geomtime, ic_func, on_initial)
bc = dde.icbc.DirichletBC(geomtime, bc_func, on_boundary)

print("Initial and boundary conditions set.")

# ============================================================
# БЛОК 5: Данные и архитектура нейросети
# ============================================================
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc],
    num_domain=2000,      # меньше точек = быстрее
    num_boundary=150,
    num_initial=150,
)

# Сеть поменьше: 5 слоев по 30 нейронов (было 8x40)
net = dde.nn.FNN(
    layer_sizes=[2] + [30] * 5 + [1],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

model = dde.Model(data, net)
print("Neural network created: 5 layers x 30 neurons, tanh")

# ============================================================
# БЛОК 6: Обучение
# ============================================================
model.compile("adam", lr=1e-3)
print("\nTraining started...")
print("=" * 50)

start_time = time.time()

losshistory, train_state = model.train(
    iterations=10000,          # меньше эпох
    display_every=500,         # чаще вывод
    model_save_path="burgers_1d_pinn",
)

training_time = time.time() - start_time
print(f"\nTraining completed! Time: {training_time:.2f} sec")

# ============================================================
# БЛОК 7: Визуализация Loss-функции
# ============================================================
# Конвертируем losshistory в numpy-массив
loss_train = np.array(losshistory.loss_train)

loss_pde = loss_train[:, 0]   # PDE loss
loss_ic = loss_train[:, 1]    # IC loss
loss_bc = loss_train[:, 2]    # BC loss
loss_total = np.sum(loss_train, axis=1)

epochs = np.arange(len(loss_total)) * 500  # display_every=500

plt.figure(figsize=(10, 6))
plt.semilogy(epochs, loss_total, 'k-', linewidth=2, label='Total Loss')
plt.semilogy(epochs, loss_pde, 'r--', linewidth=1.5, label='PDE Loss')
plt.semilogy(epochs, loss_ic, 'g--', linewidth=1.5, label='IC Loss')
plt.semilogy(epochs, loss_bc, 'b--', linewidth=1.5, label='BC Loss')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (log scale)', fontsize=12)
plt.title('PINN Training Loss -- 1D Burgers Equation', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_path = os.path.join(os.getcwd(), 'burgers_1d_pinn_loss.png')
plt.savefig(loss_path, dpi=150)
print(f"Loss plot saved: {loss_path}")

# ============================================================
# БЛОК 8: Визуализация решения
# ============================================================
print("\nGenerating solution plots...")

# Сетка для предсказания
x_grid = np.linspace(-1, 1, 256)
t_grid = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_grid, t_grid)

# Предсказание PINN
x_flat = X.flatten()[:, None]
t_flat = T.flatten()[:, None]
u_pred = model.predict(np.hstack([x_flat, t_flat]))
U_pred = u_pred.reshape(X.shape)

# Точное решение
print("Calculating exact solution...")
U_exact = exact_solution(X, T)
print("Exact solution calculated.")

# --- График 1: 3D-поверхности ---
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, T, U_pred, cmap='viridis', edgecolor='none', alpha=0.9)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('t', fontsize=11)
ax1.set_zlabel('u(x,t)', fontsize=11)
ax1.set_title('PINN Solution', fontsize=13)
ax1.view_init(elev=25, azim=-60)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, T, U_exact, cmap='viridis', edgecolor='none', alpha=0.9)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('t', fontsize=11)
ax2.set_zlabel('u(x,t)', fontsize=11)
ax2.set_title('Exact Solution', fontsize=13)
ax2.view_init(elev=25, azim=-60)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.suptitle('1D Burgers Equation: PINN vs Exact', fontsize=15, y=1.02)
plt.tight_layout()

plot3d_path = os.path.join(os.getcwd(), 'burgers_1d_pinn_3d.png')
plt.savefig(plot3d_path, dpi=150, bbox_inches='tight')
print(f"3D plot saved: {plot3d_path}")

# --- График 2: Карта ошибки ---
plt.figure(figsize=(10, 6))
error = np.abs(U_pred - U_exact)
im = plt.pcolormesh(X, T, error, shading='auto', cmap='hot')
plt.colorbar(im, label='Absolute Error |u_pred - u_exact|')
plt.xlabel('x', fontsize=12)
plt.ylabel('t', fontsize=12)
plt.title('PINN Absolute Error Map', fontsize=14)
plt.tight_layout()

error_path = os.path.join(os.getcwd(), 'burgers_1d_pinn_error.png')
plt.savefig(error_path, dpi=150)
print(f"Error map saved: {error_path}")

# --- График 3: Срезы по времени ---
t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for i, t_val in enumerate(t_slices):
    ax = axes[i]
    t_idx = np.argmin(np.abs(t_grid - t_val))
    
    ax.plot(x_grid, U_exact[t_idx, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x_grid, U_pred[t_idx, :], 'r--', linewidth=2, label='PINN')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('u(x,t)', fontsize=10)
    ax.set_title(f't = {t_val}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.2, 1.2])

# Скрываем пустой subplot если есть
if len(t_slices) < len(axes):
    for j in range(len(t_slices), len(axes)):
        axes[j].axis('off')

plt.suptitle('PINN vs Exact: Time Slices', fontsize=15)
plt.tight_layout()

slices_path = os.path.join(os.getcwd(), 'burgers_1d_pinn_slices.png')
plt.savefig(slices_path, dpi=150)
print(f"Time slices saved: {slices_path}")

# --- График 4: Финальный срез t=1.0 ---
plt.figure(figsize=(9, 5))
t_idx = -1
plt.plot(x_grid, U_exact[t_idx, :], 'b-', linewidth=2.5, label='Exact')
plt.plot(x_grid, U_pred[t_idx, :], 'r--', linewidth=2.5, label='PINN')
plt.fill_between(x_grid, U_exact[t_idx, :], U_pred[t_idx, :], 
                 alpha=0.3, color='red', label='Error region')
plt.xlabel('x', fontsize=12)
plt.ylabel('u(x, t=1.0)', fontsize=12)
plt.title('Shock Front at t = 1.0: PINN vs Exact', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

final_path = os.path.join(os.getcwd(), 'burgers_1d_pinn_final.png')
plt.savefig(final_path, dpi=150)
print(f"Final slice saved: {final_path}")

# ============================================================
# БЛОК 9: Метрики точности
# ============================================================
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(U_exact.flatten(), U_pred.flatten())
mae = mean_absolute_error(U_exact.flatten(), U_pred.flatten())
rel_l2 = np.linalg.norm(U_exact - U_pred) / np.linalg.norm(U_exact)

print("\n" + "=" * 50)
print("ACCURACY METRICS")
print("=" * 50)
print(f"MSE (Mean Squared Error):        {mse:.6e}")
print(f"MAE (Mean Absolute Error):       {mae:.6e}")
print(f"Relative L2 Error:               {rel_l2:.6e}")
print(f"Training time:                   {training_time:.2f} seconds")
print("=" * 50)

# Показываем все графики
plt.show()

print("\nDone! All plots saved in:", os.getcwd())