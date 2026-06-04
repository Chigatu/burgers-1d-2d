# -*- coding: utf-8 -*-
"""Generate reference solution for 1D Burgers using analytical Cole-Hopf"""
import numpy as np
from scipy.special import iv
import os

nu = 0.01 / np.pi

def burgers_exact(x, t, N=100):
    """
    Exact 1D Burgers solution via Cole-Hopf for u(x,0) = -sin(pi*x).
    Stable implementation with clipping.
    """
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)
    
    result = np.zeros((len(t), len(x)))
    
    for j, tj in enumerate(t):
        if tj < 1e-12:
            result[j] = -np.sin(np.pi * x)
            continue
        
        # Вычисляем коэффициенты с защитой от переполнения
        # Используем асимптотику iv ~ exp(x)/sqrt(2*pi*x) для больших аргументов
        arg = 1.0 / (2.0 * nu)
        
        # Логарифмический трюк для устойчивости
        log_iv0 = np.log(iv(0, arg) + 1e-300)
        
        numerator = np.zeros_like(x)
        denominator = np.zeros_like(x)
        
        for n in range(1, N + 1):
            # Устойчивое вычисление отношения iv(n,arg)/iv(0,arg)
            log_iv_n = np.log(iv(n, arg) + 1e-300)
            log_coef = log_iv_n - log_iv0
            
            coef = np.exp(log_coef)
            if np.isnan(coef) or np.isinf(coef):
                coef = 1.0  # асимптотика для больших n
            
            a_n = 2.0 * ((-1.0) ** n) * coef
            
            exp_term = np.exp(-nu * n**2 * np.pi**2 * tj)
            sin_term = np.sin(n * np.pi * x)
            
            term = a_n * sin_term * exp_term
            numerator += n * term
            denominator += term
        
        # Защита от деления на ноль
        denom = 1.0 + 2.0 * denominator
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        
        result[j] = 2.0 * nu * np.pi * numerator / denom
    
    return result

# Сетка
x_grid = np.linspace(-1, 1, 512)
t_grid = np.linspace(0, 1, 100)

print("Computing reference solution...")
u = burgers_exact(x_grid, t_grid, N=100)

# Проверка
print(f"u shape: {u.shape}")
print(f"t=0.0: min={u[0].min():.4f}, max={u[0].max():.4f}")
print(f"t=0.5: min={u[50].min():.4f}, max={u[50].max():.4f}")
print(f"t=1.0: min={u[-1].min():.4f}, max={u[-1].max():.4f}")

# Проверка на NaN
if np.any(np.isnan(u)):
    print("WARNING: NaN detected!")
else:
    print("OK: No NaN values.")

np.savez('reference_1d.npz', x=x_grid, t=t_grid, u=u)
print("\nReference solution saved to reference_1d.npz")