# -*- coding: utf-8 -*-
"""
1D Burgers Equation - PINN + RAR (Residual-based Adaptive Refinement)
u_t + u * u_x - nu * u_xx = 0
RAR: automatic point refinement near shock front
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
from scipy.special import iv
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Libraries loaded successfully!")

# ============================================================
# БЛОК 1: Параметры
# ============================================================
nu = 0.01 / np.pi
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ============================================================
# БЛОК 2: Точное решение
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

# ============================================================
# БЛОК 3: PDE
# ============================================================
def pde(x, u):
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t + u * du_x - nu * du_xx

# ============================================================
# БЛОК 4: Начальные и граничные условия
# ============================================================
def ic_func(x):
    return -np.sin(np.pi * x[:, 0:1])

ic = dde.icbc.IC(geomtime, ic_func, lambda _, on: on)
bc = dde.icbc.DirichletBC(geomtime, lambda x: np.zeros((len(x), 1)), lambda _, on: on)

# ============================================================
# БЛОК 5: Начальные данные для обучения
# ============================================================
data = dde.data.TimePDE(
    geomtime, pde, [ic, bc],
    num_domain=512,     # начинаем с малого числа точек
    num_boundary=100,
    num_initial=100,
)

# ============================================================
# БЛОК 6: Сеть
# ============================================================
net = dde.nn.FNN(
    layer_sizes=[2] + [50] * 6 + [1],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# ============================================================
# БЛОК 7: Обучение с RAR
# ============================================================
print("\n" + "="*60)
print("TRAINING PINN + RAR")
print("="*60)

start_time = time.time()

# Этап 1: Предварительное обучение (грубое решение)
print("\n--- Stage 1: Initial training (5000 epochs) ---")
losshistory, train_state = model.train(iterations=5000, display_every=500)

# Этап 2: RAR — добавляем точки где невязка максимальна
print("\n--- Stage 2: RAR refinement ---")
# Сетка для проверки невязки
x_test = np.linspace(-1, 1, 200)
t_test = np.linspace(0, 1, 100)
X_test, T_test = np.meshgrid(x_test, t_test)
test_points = np.hstack([X_test.flatten()[:, None], T_test.flatten()[:, None]])

# Вычисляем невязку
residuals = np.abs(model.predict(test_points, operator=pde))

# Сортируем точки по величине невязки
sorted_idx = np.argsort(residuals.flatten())[::-1]  # по убыванию

# Берём top-500 точек с наибольшей невязкой
top_k = 500
new_points = test_points[sorted_idx[:top_k]]

print(f"Added {top_k} new collocation points at locations of highest residual")

# Визуализация распределения точек (до и после RAR)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Исходные точки (равномерные)
orig_points = geomtime.random_points(512)
axes[0].scatter(orig_points[:, 0], orig_points[:, 1], c='blue', s=10, alpha=0.6)
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
axes[0].set_title('Before RAR: Uniform Points (512)')
axes[0].set_xlim([-1, 1])
axes[0].set_ylim([0, 1])

# Точки после RAR (все вместе)
all_points = np.vstack([orig_points, new_points])
axes[1].scatter(orig_points[:, 0], orig_points[:, 1], c='blue', s=10, alpha=0.3, label='Original')
axes[1].scatter(new_points[:, 0], new_points[:, 1], c='red', s=15, alpha=0.8, label='RAR Added')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
axes[1].set_title(f'After RAR: Added {top_k} Points at High Residual')
axes[1].legend()
axes[1].set_xlim([-1, 1])
axes[1].set_ylim([0, 1])

rar_dist_path = os.path.join(os.getcwd(), 'burgers_1d_rar_distribution.png')
plt.savefig(rar_dist_path, dpi=150, bbox_inches='tight')
print(f"RAR point distribution saved: {rar_dist_path}")

# Обновляем данные — добавляем новые точки
data = dde.data.TimePDE(
    geomtime, pde, [ic, bc],
    num_domain=512,
    num_boundary=100,
    num_initial=100,
    anchors=new_points,  # ключевой параметр RAR!
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-4)  # уменьшаем learning rate для тонкой настройки

# Этап 3: Дообучение с новыми точками
print("\n--- Stage 3: Fine-tuning with RAR points (10000 epochs) ---")
losshistory, train_state = model.train(
    iterations=10000,
    display_every=500,
    model_save_path="burgers_1d_rar",
)

training_time = time.time() - start_time
print(f"\nTotal training time: {training_time:.2f} sec")

# ============================================================
# БЛОК 8: Визуализация Loss
# ============================================================
loss_train = np.array(losshistory.loss_train)
epochs = np.arange(len(loss_train)) * 500

plt.figure(figsize=(10, 6))
plt.semilogy(epochs, np.sum(loss_train, axis=1), 'k-', linewidth=2, label='Total Loss')
plt.semilogy(epochs, loss_train[:, 0], 'r--', linewidth=1, label='PDE Loss')
plt.semilogy(epochs, loss_train[:, 1], 'g--', linewidth=1, label='IC Loss')
plt.semilogy(epochs, loss_train[:, 2], 'b--', linewidth=1, label='BC Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (log scale)')
plt.title('PINN + RAR Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

rar_loss_path = os.path.join(os.getcwd(), 'burgers_1d_rar_loss.png')
plt.savefig(rar_loss_path, dpi=150)
print(f"RAR loss plot saved: {rar_loss_path}")

# ============================================================
# БЛОК 9: Визуализация решения
# ============================================================
x_grid = np.linspace(-1, 1, 256)
t_grid = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_grid, t_grid)

u_pred = model.predict(np.hstack([X.flatten()[:, None], T.flatten()[:, None]]))
U_pred = u_pred.reshape(X.shape)
U_exact = exact_solution(X, T)

# 3D сравнение
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, T, U_pred, cmap='coolwarm', edgecolor='none', alpha=0.9)
ax1.set_xlabel('x'); ax1.set_ylabel('t'); ax1.set_zlabel('u')
ax1.set_title('PINN + RAR Solution')
ax1.view_init(25, -60)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, T, U_exact, cmap='coolwarm', edgecolor='none', alpha=0.9)
ax2.set_xlabel('x'); ax2.set_ylabel('t'); ax2.set_zlabel('u')
ax2.set_title('Exact Solution')
ax2.view_init(25, -60)
plt.tight_layout()

rar_3d_path = os.path.join(os.getcwd(), 'burgers_1d_rar_3d.png')
plt.savefig(rar_3d_path, dpi=150)
print(f"RAR 3D plot saved: {rar_3d_path}")

# Карта ошибки
error = np.abs(U_pred - U_exact)
plt.figure(figsize=(10, 6))
plt.pcolormesh(X, T, error, shading='auto', cmap='hot')
plt.colorbar(label='Absolute Error')
plt.xlabel('x'); plt.ylabel('t')
plt.title('PINN + RAR Absolute Error')
plt.tight_layout()

rar_error_path = os.path.join(os.getcwd(), 'burgers_1d_rar_error.png')
plt.savefig(rar_error_path, dpi=150)
print(f"RAR error map saved: {rar_error_path}")

# Срезы по времени
t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for i, t_val in enumerate(t_slices):
    t_idx = np.argmin(np.abs(t_grid - t_val))
    axes[i].plot(x_grid, U_exact[t_idx, :], 'b-', linewidth=2, label='Exact')
    axes[i].plot(x_grid, U_pred[t_idx, :], 'r--', linewidth=2, label='PINN+RAR')
    axes[i].set_xlabel('x'); axes[i].set_ylabel('u')
    axes[i].set_title(f't = {t_val}')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim([-1.2, 1.2])
for j in range(len(t_slices), len(axes)):
    axes[j].axis('off')
plt.suptitle('PINN + RAR vs Exact: Time Slices', fontsize=14)
plt.tight_layout()

rar_slices_path = os.path.join(os.getcwd(), 'burgers_1d_rar_slices.png')
plt.savefig(rar_slices_path, dpi=150)
print(f"RAR slices saved: {rar_slices_path}")

# ============================================================
# БЛОК 10: Сравнение PINN vs PINN+RAR (на одном графике)
# ============================================================
# Загружаем результаты PINN (если файл есть)
pinn_pred_path = os.path.join(os.getcwd(), '..', 'burgers_1d_pinn-15000.weights.h5')
# Рисуем сравнение в конечный момент
plt.figure(figsize=(10, 6))
t_idx = -1
plt.plot(x_grid, U_exact[t_idx, :], 'b-', linewidth=2.5, label='Exact')
plt.plot(x_grid, U_pred[t_idx, :], 'r--', linewidth=2.5, label='PINN + RAR')
plt.xlabel('x'); plt.ylabel('u(x, t=1.0)')
plt.title('Shock Front: PINN+RAR vs Exact at t=1.0')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()

rar_final_path = os.path.join(os.getcwd(), 'burgers_1d_rar_final.png')
plt.savefig(rar_final_path, dpi=150)
print(f"RAR final slice saved: {rar_final_path}")

# ============================================================
# БЛОК 11: Метрики
# ============================================================
mse = mean_squared_error(U_exact.flatten(), U_pred.flatten())
mae = mean_absolute_error(U_exact.flatten(), U_pred.flatten())
rel_l2 = np.linalg.norm(U_exact - U_pred) / np.linalg.norm(U_exact)

print("\n" + "="*60)
print("PINN + RAR ACCURACY METRICS")
print("="*60)
print(f"MSE:              {mse:.6e}")
print(f"MAE:              {mae:.6e}")
print(f"Relative L2:      {rel_l2:.6e}")
print(f"Training time:    {training_time:.2f} sec")
print("="*60)

plt.show()
print("\nDone! All RAR plots saved.")