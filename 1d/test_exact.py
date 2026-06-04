# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import iv

nu = 0.01 / np.pi

def exact_solution(x, t):
    """Load precomputed reference solution"""
    import os
    data = np.load(os.path.join(os.path.dirname(__file__), 'reference_1d.npz'))
    x_ref = data['x']
    t_ref = data['t']
    u_ref = data['u']
    
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((t_ref, x_ref), u_ref, bounds_error=False, fill_value=0)
    
    # Создаём сетку
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)
    X, T = np.meshgrid(x, t)
    points = np.stack([T.flatten(), X.flatten()], axis=-1)
    result = interp(points).reshape(len(t), len(x))
    return result.squeeze()

# Тест
x = np.linspace(-1, 1, 10)
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    u = exact_solution(x, t)
    print(f"t={t}: u = {u[:3]}... min={u.min():.4f}, max={u.max():.4f}")