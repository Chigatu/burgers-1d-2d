# Нейросетевое решение уравнения Бюргерса

Методы: **PINN**, **PINN+RAR**, **DeepONet** для 1D и 2D задач

---

## Физические задачи

| | 1D | 2D |
|---|-----|-----|
| **Что моделирует** | Ударная волна в трубе | Двумерная турбулентность |
| **Начальное условие** | `u(x,0) = -sin(πx)` | Набор вихрей (гауссианы) |
| **Аналогия** | Автомобильная пробка | Вихри в плоском слое жидкости |

---

## Структура проекта

```
burgers_project/
├── 1d/                         # 1D Burgers equation
│   ├── burgers_pinn_1d.py      # PINN
│   ├── burgers_rar_1d.py       # PINN + RAR
│   ├── burgers_deeponet_1d.py  # DeepONet
│   ├── generate_reference_1d.py
│   ├── generate_deeponet_data.py
│   └── *.png                   # визуализации
├── 2d/                         # 2D Burgers equation
│   ├── burgers_pinn_2d.py      # PINN
│   ├── burgers_deeponet_2d_demo.py  # DeepONet MINI
│   ├── generate_reference_2d.py
│   └── *.png                   # визуализации
├── plot_comparison.py          # сравнительные графики
├── create_presentation.py      # генератор презентации
├── *.pptx                      # готовая презентация (25 слайдов)
└── README.md
```

---

## Результаты

| Модель | Отн. L2 ↓ | Время | Данные? |
|--------|-----------|-------|---------|
| PINN 1D | 8.27 | 117с | Нет |
| PINN+RAR 1D | 8.37 | 166с | Нет |
| **DeepONet 1D** | **0.62** ★ | 310с | Да |
| PINN 2D | 0.53 | 202с | Нет |
| **DeepONet 2D MINI** | **0.28** ★★ | **69с** | Да |

**★ Лучший в 1D:** DeepONet (в 13 раз точнее PINN)
**★★ Абсолютный победитель:** DeepONet 2D MINI

---

## Методы

### PINN (Physics-Informed Neural Network)
Полносвязная сеть, обучаемая на физических уравнениях. Не требует данных — только уравнение, начальные и граничные условия. Производные вычисляются автоматическим дифференцированием.

### PINN + RAR (Residual-based Adaptive Refinement)
Автоматически добавляет коллокационные точки в областях с максимальной невязкой PDE. Сеть фокусируется на ударных фронтах.

### DeepONet (Deep Operator Network)
Операторная сеть из двух подсетей (Branch + Trunk). Учит отображение `G: u(x,0) → u(x,t)`. После обучения предсказывает решение для нового начального условия без переобучения.

---

## Установка и запуск

*```bash*
pip install deepxde tensorflow matplotlib numpy scipy scikit-learn python-pptx
```

### 1D PINN:
*```bash*
cd 1d
python generate_reference_1d.py
python burgers_pinn_1d.py
```

### 1D DeepONet:
*```bash*
cd 1d
python generate_deeponet_data.py
python burgers_deeponet_1d.py
```

### 2D PINN:
*```bash*
cd 2d
python generate_reference_2d.py
python burgers_pinn_2d.py
```

### Сравнительные графики:
*```bash*
python plot_comparison.py
```

### Генерация презентации:
*```bash*
python create_presentation.py
```

---

## Автор

**Аршинов И.А.**
Группа М8О-105СВ-2025

Московский авиационный институт (МАИ)
Кафедра 806 «Вычислительная математика и программирование»

Курс: «Физически информированное машинное обучение»
Преподаватель: Стрижак С.В.

Москва, 2026