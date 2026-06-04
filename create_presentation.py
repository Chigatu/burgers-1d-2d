# -*- coding: utf-8 -*-
"""
Auto-generate PowerPoint presentation:
"Нейросетевое решение уравнения Бюргерса методами PINN, PINN+RAR и DeepONet"
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

# Пути к графикам
BASE = os.path.dirname(os.path.abspath(__file__))
D1 = os.path.join(BASE, '1d')
D2 = os.path.join(BASE, '2d')

prs = Presentation()
prs.slide_width = Inches(13.333)  # 16:9
prs.slide_height = Inches(7.5)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def add_slide(prs):
    """Add blank slide"""
    layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(layout)

def add_textbox(slide, left, top, width, height):
    """Add empty textbox"""
    return slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))

def add_centered_text(tf, text, font_size=14, bold=False, color=None, space_after=4):
    """Add centered paragraph to text frame"""
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.alignment = PP_ALIGN.CENTER
    if color:
        p.font.color.rgb = color
    p.space_after = Pt(space_after)
    return p

def add_left_text(tf, text, font_size=14, bold=False, color=None, space_after=4):
    """Add left-aligned paragraph"""
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.alignment = PP_ALIGN.LEFT
    if color:
        p.font.color.rgb = color
    p.space_after = Pt(space_after)
    return p

def add_title(slide, text, left=0.5, top=0.3, width=12, height=0.8):
    """Add title text"""
    txBox = add_textbox(slide, left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
    p.alignment = PP_ALIGN.LEFT
    return txBox

def add_body(slide, text, left=0.5, top=1.5, width=12, height=5.5, font_size=18):
    """Add body text with bullet points"""
    txBox = add_textbox(slide, left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    lines = text.strip().split('\n')
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(4)
    return txBox

def add_code_block(slide, code_text, left=0.5, top=1.5, width=5.5, height=4.0):
    """Add code block with dark background"""
    from pptx.util import Pt
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x1e, 0x1e, 0x2e)
    shape.line.fill.background()
    
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(8)
    tf.margin_top = Pt(6)
    
    for i, line in enumerate(code_text.strip().split('\n')):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(9)
        p.font.name = 'Consolas'
        p.font.color.rgb = RGBColor(0xcd, 0xd6, 0xf4)
        p.space_after = Pt(1)
    
    return shape

def add_image_safe(slide, path, left, top, width, height=None):
    """Add image if exists, otherwise placeholder"""
    if os.path.exists(path):
        if height:
            return slide.shapes.add_picture(path, Inches(left), Inches(top), Inches(width), Inches(height))
        else:
            return slide.shapes.add_picture(path, Inches(left), Inches(top), Inches(width))
    else:
        print(f"  WARNING: Image not found: {path}")
        return None

def add_slide_number(slide, num):
    """Add slide number in bottom-right"""
    txBox = add_textbox(slide, 12.5, 7.0, 0.7, 0.4)
    p = txBox.text_frame.paragraphs[0]
    p.text = str(num)
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    p.alignment = PP_ALIGN.RIGHT

# ============================================================
# СЛАЙД 1: ТИТУЛЬНЫЙ (по твоему шаблону)
# ============================================================
slide = add_slide(prs)

# Верхняя часть: министерство
txBox = add_textbox(slide, 1.0, 0.3, 11.3, 1.5)
tf = txBox.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = 'МИНИСТЕРСТВО ОБРАЗОВАНИЯ И НАУКИ РОССИЙСКОЙ ФЕДЕРАЦИИ'
p.font.size = Pt(11)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER
p.space_after = Pt(6)

add_centered_text(tf, 'ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ БЮДЖЕТНОЕ ОБРАЗОВАТЕЛЬНОЕ', font_size=11, bold=True)
add_centered_text(tf, 'УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ', font_size=11, bold=True)
add_centered_text(tf, '«МОСКОВСКИЙ АВИАЦИОННЫЙ ИНСТИТУТ', font_size=12, bold=True)
add_centered_text(tf, '(НАЦИОНАЛЬНЫЙ ИССЛЕДОВАТЕЛЬСКИЙ УНИВЕРСИТЕТ)» (МАИ)', font_size=12, bold=True, space_after=8)

add_centered_text(tf, 'Кафедра 806', font_size=13, bold=True)
add_centered_text(tf, '«Вычислительная математика и программирование»', font_size=12, bold=True, space_after=12)

# Тема работы
add_centered_text(tf, '', font_size=6)  # spacer
add_centered_text(tf, 'Отчёт по лабораторной работе', font_size=14, bold=True, space_after=4)
add_centered_text(tf, '«Нейросетевое решение уравнения Бюргерса', font_size=16, bold=True)
add_centered_text(tf, 'методами PINN, PINN+RAR и DeepONet', font_size=16, bold=True)
add_centered_text(tf, 'для 1D и 2D задач гидродинамики»', font_size=14, bold=True, space_after=10)

add_centered_text(tf, 'по курсу', font_size=12)
add_centered_text(tf, '«Физически информированное машинное обучение»', font_size=13, bold=True, space_after=14)

# Преподаватель и автор
add_left_text(tf, 'преподаватель: Стрижак С.В.', font_size=13, space_after=6)
add_left_text(tf, 'выполнила: Аршинов И.А.', font_size=13, space_after=4)
add_left_text(tf, 'группа: М8О-105СВ-2025', font_size=13, space_after=14)

add_centered_text(tf, 'Москва, 2026', font_size=13, bold=True)

add_slide_number(slide, 1)

# ============================================================
# СЛАЙД 2: Постановка задачи
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Постановка задачи')
add_body(slide, 
    '• Решить одномерное и двумерное уравнение Бюргерса\n'
    '  (модель нелинейных волн и ударных фронтов)\n\n'
    '• Применить три нейросетевых метода:\n'
    '  1. PINN — физико-информированная сеть (без данных)\n'
    '  2. PINN + RAR — адаптивное уточнение коллокационных точек\n'
    '  3. DeepONet — операторная сеть (обучение на датасете)\n\n'
    '• Провести сравнительный анализ:\n'
    '  – Точность (Relative L2 Error)\n'
    '  – Время обучения\n'
    '  – Требования к данным\n\n'
    '• Визуализировать ударные волны (1D) и вихревые структуры (2D)',
    font_size=17)
add_slide_number(slide, 2)

# ============================================================
# СЛАЙД 3: Цель работы
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Цель работы')
add_body(slide,
    '• Исследовать физико-информированные нейронные сети (PINN)\n'
    '  как инструмент решения дифференциальных уравнений в частных производных\n\n'
    '• Изучить метод адаптивного уточнения RAR\n'
    '  для повышения точности в областях больших градиентов\n\n'
    '• Реализовать операторную сеть DeepONet\n'
    '  как суррогатный решатель для мгновенных предсказаний\n\n'
    '• Провести сравнительный анализ методов\n'
    '  по точности, времени обучения и зависимости от данных\n\n'
    '• Подготовить набор визуализаций для демонстрации результатов',
    font_size=17)
add_slide_number(slide, 3)

# ============================================================
# СЛАЙД 4: Актуальность
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Актуальность')
add_body(slide,
    '• Уравнение Бюргерса — простейшая модель, объединяющая\n'
    '  нелинейный перенос и вязкую диффузию\n\n'
    '• Применяется в:\n'
    '  – Газодинамике (ударные волны в трубах, сопла)\n'
    '  – Теории дорожного трафика (модель пробки)\n'
    '  – Нелинейной акустике (звуковые волны большой амплитуды)\n'
    '  – Турбулентности (каскад энергии в 2D)\n\n'
    '• Классические сеточные методы требуют мелкой сетки\n'
    '  в зонах ударных фронтов → высокие вычислительные затраты\n\n'
    '• PINN и DeepONet — бессеточные нейросетевые подходы:\n'
    '  – Не требуют построения расчётной сетки\n'
    '  – DeepONet заменяет итеративный солвер прямой прогонкой сети',
    font_size=16)
add_slide_number(slide, 4)

# ============================================================
# СЛАЙД 5: Уравнение Бюргерса
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Уравнение Бюргерса')
add_body(slide,
    'ОДНОМЕРНЫЙ СЛУЧАЙ (1D):\n'
    '  ∂u/∂t + u · ∂u/∂x = ν · ∂²u/∂x²\n'
    '  x ∈ [−1, 1],  t ∈ [0, 1],  ν = 0.01/π ≈ 0.00318\n\n'
    'ДВУМЕРНЫЙ СЛУЧАЙ (2D):\n'
    '  ∂u/∂t + u · ∂u/∂x + u · ∂u/∂y = ν · (∂²u/∂x² + ∂²u/∂y²)\n'
    '  (x, y) ∈ [−1, 1]²,  t ∈ [0, 1]\n\n'
    'где:\n'
    '  u — скорость потока (1D) или скалярное поле (2D)\n'
    '  ν — кинематическая вязкость\n'
    '  u·u_x — нелинейный конвективный перенос\n'
    '  ν·u_xx — вязкая диффузия',
    font_size=17)
add_slide_number(slide, 5)

# ============================================================
# СЛАЙД 6: Физические задачи
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Физические задачи')
add_body(slide,
    '1D — УДАРНАЯ ВОЛНА В ТРУБЕ:\n'
    '• Газ в длинной узкой трубе, резкий перепад давления\n'
    '• Начальная синусоида u(x,0) = −sin(πx) опрокидывается в «пилу»\n'
    '• Ударный фронт со временем размывается вязкостью\n'
    '• Аналогия: автомобильная пробка на дороге\n\n'
    '2D — ДВУМЕРНАЯ ТУРБУЛЕНТНОСТЬ:\n'
    '• Плоский слой жидкости с несколькими вихрями\n'
    '• Вихри взаимодействуют, дробятся на мелкие структуры\n'
    '• Энергия каскадом переходит в мелкие масштабы и диссипирует\n'
    '• Начальное условие: суперпозиция гауссиан разного знака',
    top=1.2, height=2.8, font_size=16)

add_image_safe(slide, os.path.join(D2, 'burgers_2d_reference.png'),
               left=1.5, top=4.2, width=10.0, height=2.8)
add_slide_number(slide, 6)

# ============================================================
# СЛАЙД 7: Метод 1 — PINN
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Метод 1: PINN (Physics-Informed Neural Network)')
add_body(slide,
    'ПРИНЦИП РАБОТЫ:\n'
    '• Полносвязная нейронная сеть аппроксимирует решение u(x,t)\n'
    '• Вход: пространственно-временные координаты\n'
    '• Выход: значение функции u в этой точке\n\n'
    'ФУНКЦИЯ ПОТЕРЬ:\n'
    '  Loss = MSE_PDE + MSE_начало + MSE_границы\n\n'
    '• Производные вычисляются автоматическим дифференцированием\n'
    '• Данные НЕ требуются — только уравнение и граничные условия!',
    top=1.2, left=6.5, width=6, height=4.5, font_size=14)

add_code_block(slide, 
    '''def pde(x, u):
    """1D Burgers equation residual"""
    du_x = dde.grad.jacobian(u, x, i=0, j=0)   # du/dx
    du_t = dde.grad.jacobian(u, x, i=0, j=1)   # du/dt
    du_xx = dde.grad.hessian(u, x, i=0, j=0)   # d²u/dx²

    residual = du_t + u * du_x - nu * du_xx
    return residual
# nu = 0.01 / pi — вязкость
# Автоградиент вычисляет производные по графу вычислений''',
    left=0.5, top=1.2, width=5.5, height=3.5)

add_code_block(slide,
    '''# Начальные и граничные условия
ic = dde.icbc.IC(
    geomtime,
    lambda x: -sin(pi * x[:, 0:1]),  # u(x,0)
    lambda _, on: on
)
bc = dde.icbc.DirichletBC(
    geomtime,
    lambda x: zeros_like(x[:, 0:1]), # u=0
    lambda _, on: on
)''',
    left=0.5, top=4.9, width=5.5, height=2.3)
add_slide_number(slide, 7)

# ============================================================
# СЛАЙД 8: Метод 2 — PINN + RAR
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Метод 2: PINN + RAR')
add_body(slide,
    'ПРОБЛЕМА ОБЫЧНОГО PINN:\n'
    '• Равномерные коллокационные точки неэффективны\n'
    '• Ударный фронт требует больше точек, чем гладкие области\n\n'
    'RAR (RESIDUAL-BASED ADAPTIVE REFINEMENT):\n'
    '1. Обучаем PINN на равномерной сетке (грубое решение)\n'
    '2. Вычисляем невязку PDE на тестовой сетке\n'
    '3. Выбираем top-K точек с максимальной |невязкой|\n'
    '4. Добавляем их через anchors = new_points\n'
    '5. Переобучаем — сеть фокусируется на ударном фронте!',
    top=1.2, left=0.5, width=6, height=4.5, font_size=14)

add_code_block(slide,
    '''# RAR: добавляем точки с max невязкой
residuals = abs(model.predict(
    test_points, operator=pde))
sorted_idx = argsort(residuals)[::-1]
new_points = test_points[sorted_idx[:500]]

# Обновляем данные — anchors!
data = dde.data.TimePDE(
    geomtime, pde, [ic, bc],
    num_domain=512,
    anchors=new_points,  # <-- RAR!
)''',
    left=7.0, top=1.2, width=5.5, height=3.5)

add_image_safe(slide, os.path.join(D1, 'burgers_1d_rar_distribution.png'),
               left=1.0, top=5.5, width=11.0, height=1.8)
add_slide_number(slide, 8)

# ============================================================
# СЛАЙД 9: Метод 3 — DeepONet
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Метод 3: DeepONet (Deep Operator Network)')
add_body(slide,
    'ИДЕЯ: Учим оператор G, отображающий начальное условие в решение:\n'
    '  G : u(x, 0) → u(x, t)  для всех x, t\n\n'
    'АРХИТЕКТУРА (две подсети):\n'
    '• Branch Net: принимает дискретные значения u₀(x) → эмбеддинг b\n'
    '• Trunk Net: принимает координаты (x, t) → базис t\n'
    '• Выход: скалярное произведение <b, t> + bias\n\n'
    'ПОСЛЕ ОБУЧЕНИЯ:\n'
    '• Предсказание для нового u₀ без переобучения!\n'
    '• Одна прямая прогонка сети = доли секунды',
    top=1.2, left=6.5, width=6, height=4.5, font_size=14)

add_code_block(slide,
    '''# DeepONet: операторная сеть
net = dde.nn.DeepONetCartesianProd(
    layer_sizes_branch=[
        128, 128, 128, 100  # u0 -> embedding
    ],
    layer_sizes_trunk=[
        2, 128, 128, 100     # (x,t) -> basis
    ],
    activation="relu",
    kernel_initializer="Glorot normal",
)
# Branch: 128 входов (сетка u0)
# Trunk: 2 входа (x, t)
# Выход: dot(branch, trunk)''',
    left=0.5, top=1.2, width=5.5, height=3.5)

add_code_block(slide,
    '''# Данные для DeepONet
data = dde.data.Triple(
    X_train=(branch_train, trunk_train),
    y_train=sol_train,
    X_test=(branch_test, trunk_test),
    y_test=sol_test,
)
# branch: начальные условия u0(x)
# trunk: координаты (x,t)
# sol: соответствующие решения''',
    left=0.5, top=5.0, width=5.5, height=2.3)
add_slide_number(slide, 9)

# ============================================================
# СЛАЙД 10: Инструменты
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Инструменты и структура проекта')
add_body(slide,
    'ТЕХНОЛОГИИ:\n'
    '• Python 3.12, TensorFlow 2.21 (backend)\n'
    '• DeepXDE 1.15 — фреймворк для PINN и DeepONet\n'
    '• NumPy, Matplotlib, SciPy, Scikit-learn\n\n'
    'УСТАНОВКА:\n'
    '  pip install deepxde tensorflow matplotlib numpy scipy scikit-learn\n\n'
    'СТРУКТУРА ПРОЕКТА:\n'
    '  burgers_project/\n'
    '  ├── 1d/    ← PINN, RAR, DeepONet 1D (6 скриптов, 16+ графиков)\n'
    '  │   ├── burgers_pinn_1d.py\n'
    '  │   ├── burgers_rar_1d.py\n'
    '  │   ├── burgers_deeponet_1d.py\n'
    '  │   └── generate_*.py (датасеты)\n'
    '  ├── 2d/    ← PINN, DeepONet 2D (3 скрипта, 8+ графиков)\n'
    '  │   ├── burgers_pinn_2d.py\n'
    '  │   └── burgers_deeponet_2d_demo.py\n'
    '  └── plot_comparison.py ← сравнительные графики',
    font_size=14)
add_slide_number(slide, 10)

# ============================================================
# СЛАЙД 11: 1D Постановка
# ============================================================
slide = add_slide(prs)
add_title(slide, '1D Бюргерс: постановка эксперимента')
add_body(slide,
    'УРАВНЕНИЕ:\n'
    '  u_t + u · u_x = ν · u_xx\n\n'
    'ОБЛАСТЬ:\n'
    '  x ∈ [−1, 1],  t ∈ [0, 1]\n\n'
    'ВЯЗКОСТЬ:\n'
    '  ν = 0.01 / π ≈ 0.00318\n\n'
    'НАЧАЛЬНОЕ УСЛОВИЕ:\n'
    '  u(x, 0) = −sin(πx)\n\n'
    'ГРАНИЧНЫЕ УСЛОВИЯ:\n'
    '  u(−1, t) = u(1, t) = 0  (Dirichlet)',
    font_size=20)
add_slide_number(slide, 11)

# ============================================================
# СЛАЙД 12: 1D PINN — архитектура и обучение
# ============================================================
slide = add_slide(prs)
add_title(slide, '1D PINN: архитектура и обучение')

add_body(slide,
    'АРХИТЕКТУРА:\n'
    '• 5 скрытых слоёв × 30 нейронов\n'
    '• Функция активации: tanh\n'
    '• 2000 коллокационных точек + 100 начальных + 100 граничных\n\n'
    'ОБУЧЕНИЕ:\n'
    '• 10000 эпох, оптимизатор Adam\n'
    '• Learning rate: 1e-3\n'
    '• Время обучения: 117 секунд (CPU)\n'
    '• Train loss: 4.55 → 3.73×10⁻⁴',
    top=1.2, left=0.5, width=5.5, height=3.5, font_size=14)

add_code_block(slide,
    '''# Архитектура PINN
net = dde.nn.FNN(
    layer_sizes=[2] + [30] * 5 + [1],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

losshistory, train_state = model.train(
    iterations=10000,
    display_every=500,
    model_save_path="burgers_1d_pinn",
)''',
    left=6.5, top=1.2, width=6, height=3.5)

add_image_safe(slide, os.path.join(D1, 'burgers_1d_pinn_loss.png'),
               left=0.5, top=5.0, width=12.0, height=2.3)
add_slide_number(slide, 12)

# ============================================================
# СЛАЙД 13: 1D PINN — результаты
# ============================================================
slide = add_slide(prs)
add_title(slide, '1D PINN: результаты')

add_image_safe(slide, os.path.join(D1, 'burgers_1d_pinn_3d.png'),
               left=0.3, top=1.2, width=6.2, height=3.0)
add_image_safe(slide, os.path.join(D1, 'burgers_1d_pinn_error.png'),
               left=6.8, top=1.2, width=6.0, height=3.0)

add_body(slide,
    '• PINN восстанавливает ударный фронт без данных\n'
    '• Relative L2 Error = 8.27\n'
    '• Основная ошибка — в зоне формирования фронта (t ≈ 0.5)\n'
    '• Причина: мало точек в области больших градиентов',
    top=4.3, left=0.5, width=7, height=2, font_size=16)

add_image_safe(slide, os.path.join(D1, 'burgers_1d_pinn_slices.png'),
               left=0.5, top=5.8, width=12.0, height=1.5)
add_slide_number(slide, 13)

# ============================================================
# СЛАЙД 14: 1D RAR
# ============================================================
slide = add_slide(prs)
add_title(slide, '1D PINN+RAR: адаптивное уточнение')

add_body(slide,
    '• Stage 1: обучение на 512 равномерных точках (5000 эпох)\n'
    '• Stage 2: вычисление невязки → добавление 500 точек в пиках\n'
    '• Stage 3: переобучение с anchors (10000 эпох, lr=1e-4)\n\n'
    'РЕЗУЛЬТАТ:\n'
    '• Видно сгущение красных точек у ударного фронта\n'
    '• Relative L2 = 8.37 — сопоставимо с PINN\n'
    '• RAR не дал выигрыша: фронт недостаточно острый при ν=0.003',
    top=1.2, left=0.5, width=5.5, height=4, font_size=14)

add_image_safe(slide, os.path.join(D1, 'burgers_1d_rar_distribution.png'),
               left=6.5, top=1.2, width=6.5, height=3.0)
add_image_safe(slide, os.path.join(D1, 'burgers_1d_rar_loss.png'),
               left=1.0, top=5.0, width=11.0, height=2.3)
add_slide_number(slide, 14)

# ============================================================
# СЛАЙД 15: 1D DeepONet — датасет
# ============================================================
slide = add_slide(prs)
add_title(slide, '1D DeepONet: датасет')
add_body(slide,
    'ГЕНЕРАЦИЯ ДАННЫХ:\n'
    '• 1000 начальных условий: случайные суммы 2-5 гауссиан\n'
    '• Каждое решено split-step FFT методом (спектральная точность)\n'
    '• Сетка: 128 точек по x, 50 срезов по времени\n'
    '• Разделение: 800 train / 200 test\n\n'
    'ПАРАМЕТРЫ ГАУССИАН:\n'
    '• Амплитуда: a ∈ [−1.5, 1.5]\n'
    '• Центр: c ∈ [−0.6, 0.6]\n'
    '• Ширина: w ∈ [0.15, 0.5]',
    top=1.2, left=6.5, width=6, height=4, font_size=14)

add_code_block(slide,
    '''# Генерация одного сэмпла
n_gauss = randint(2, 6)
u0 = zeros(N_x)
for _ in range(n_gauss):
    a = uniform(-1.5, 1.5)
    c = uniform(-0.6, 0.6)
    w = uniform(0.15, 0.5)
    u0 += a * exp(-(x - c)**2 / w**2)

# Решение FFT-методом
u = solve_burgers_fft(u0, t_grid, nu)''',
    left=0.5, top=1.2, width=5.5, height=3.0)

add_image_safe(slide, os.path.join(D1, 'deeponet_dataset_preview.png'),
               left=0.5, top=4.5, width=12.0, height=2.8)
add_slide_number(slide, 15)

# ============================================================
# СЛАЙД 16: 1D DeepONet — результаты
# ============================================================
slide = add_slide(prs)
add_title(slide, '1D DeepONet: результаты')
add_body(slide,
    '• 20000 эпох, 310 секунд\n'
    '• Relative L2 Error = 0.62\n'
    '• В 13 раз точнее PINN (8.27 → 0.62)!\n'
    '• Предсказание для нового u₀ — доли секунды',
    top=1.2, left=6.5, width=6, height=2.5, font_size=18)

add_image_safe(slide, os.path.join(D1, 'burgers_1d_deeponet_loss.png'),
               left=0.5, top=1.2, width=5.5, height=2.5)
add_image_safe(slide, os.path.join(D1, 'burgers_1d_deeponet_test.png'),
               left=0.3, top=4.0, width=6.5, height=3.2)
add_image_safe(slide, os.path.join(D1, 'burgers_1d_deeponet_error.png'),
               left=7.0, top=4.0, width=5.8, height=3.2)
add_slide_number(slide, 16)

# ============================================================
# СЛАЙД 17: 2D Постановка
# ============================================================
slide = add_slide(prs)
add_title(slide, '2D Бюргерс: постановка эксперимента')

add_body(slide,
    'УРАВНЕНИЕ:\n'
    '  u_t + u·u_x + u·u_y = ν·(u_xx + u_yy)\n\n'
    'ОБЛАСТЬ: (x,y) ∈ [−1, 1]², t ∈ [0, 1]\n'
    'ВЯЗКОСТЬ: ν = 0.01/π\n\n'
    'НАЧАЛЬНОЕ УСЛОВИЕ:\n'
    '  3 вихря — суперпозиция гауссиан разного знака\n\n'
    'ГРАНИЦЫ: u = 0 (Dirichlet)',
    top=1.2, left=0.5, width=5.5, height=4, font_size=16)

add_code_block(slide,
    '''def pde(x, u):
    """2D Burgers residual"""
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_y = dde.grad.jacobian(u, x, i=0, j=1)
    du_t = dde.grad.jacobian(u, x, i=0, j=2)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    return (du_t + u*du_x + u*du_y
            - nu*(du_xx + du_yy))''',
    left=6.5, top=1.2, width=6, height=3.5)

add_image_safe(slide, os.path.join(D2, 'burgers_2d_reference.png'),
               left=1.5, top=5.2, width=10.0, height=2.0)
add_slide_number(slide, 17)

# ============================================================
# СЛАЙД 18: 2D PINN — результаты
# ============================================================
slide = add_slide(prs)
add_title(slide, '2D PINN: результаты')
add_body(slide,
    '• Сеть: 5 слоёв × 40 нейронов, активация tanh\n'
    '• 1500 коллокационных + 300 начальных + 300 граничных точек\n'
    '• 15000 эпох, время обучения: 202 секунды\n\n'
    '• Relative L2 Error = 0.53 — ХОРОШАЯ ТОЧНОСТЬ!\n'
    '• В 15 раз точнее, чем PINN в 1D (8.27)\n'
    '• Причина: 2D решение гладкое, без острых ударных фронтов\n'
    '  → сеть легко учит вихревые структуры',
    top=1.2, left=0.5, width=5.5, height=4, font_size=14)

add_image_safe(slide, os.path.join(D2, 'burgers_2d_pinn_loss.png'),
               left=6.5, top=1.2, width=6, height=2.8)
add_image_safe(slide, os.path.join(D2, 'burgers_2d_pinn_heatmaps.png'),
               left=0.5, top=5.0, width=12.0, height=2.3)
add_slide_number(slide, 18)

# ============================================================
# СЛАЙД 19: 2D DeepONet MINI — подход
# ============================================================
slide = add_slide(prs)
add_title(slide, '2D DeepONet MINI: упрощённая задача')
add_body(slide,
    'ПРОБЛЕМА ПОЛНОГО DEEPONET 2D:\n'
    '• 1024-мерный вход (32×32 сетка) — проклятие размерности\n'
    '• 400 примеров катастрофически мало\n'
    '• Сеть не сходится (L2 ≈ 1.0)\n\n'
    'РЕШЕНИЕ — MINI-ВЕРСИЯ:\n'
    '• 1000 сэмплов × одна гауссиана (всего 4 параметра!)\n'
    '• Сетка: 8×8×5 = 320 точек выхода (×64 сжатие)\n'
    '• PCA-сжатие входа: 1024 → 32 компоненты (98.3% дисперсии)\n'
    '• Архитектура: 32→64→64→32 (компактная)',
    top=1.2, left=0.5, width=6, height=5, font_size=14)

add_image_safe(slide, os.path.join(D2, 'burgers_2d_don_mini_loss.png'),
               left=7.0, top=1.2, width=5.8, height=3.0)
add_slide_number(slide, 19)

# ============================================================
# СЛАЙД 20: 2D DeepONet MINI — результаты
# ============================================================
slide = add_slide(prs)
add_title(slide, '2D DeepONet MINI: результаты ★')
add_body(slide,
    '• Время обучения: 69 секунд\n'
    '• Relative L2 = 0.28 ± 0.11 (50 тестовых сэмплов)\n'
    '• ЛУЧШИЙ РЕЗУЛЬТАТ СРЕДИ ВСЕХ 5 МОДЕЛЕЙ!\n'
    '• В 2 раза точнее PINN 2D (0.53 → 0.28)\n'
    '• Предсказание для нового вихря — мгновенно',
    top=1.2, left=0.5, width=6, height=3, font_size=18)

add_image_safe(slide, os.path.join(D2, 'burgers_2d_don_mini_test.png'),
               left=0.5, top=3.5, width=12.0, height=3.8)
add_slide_number(slide, 20)

# ============================================================
# СЛАЙД 21: Сравнение методов — таблица
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Сравнение всех методов')
add_body(slide,
    'МОДЕЛЬ              │ L2 ↓  │ ВРЕМЯ  │ ДАННЫЕ? │ КЛЮЧЕВОЕ\n'
    '─────────────────────┼───────┼────────┼─────────┼──────────────────\n'
    'PINN 1D             │ 8.27  │ 117с   │ Нет     │ Слабый фронт\n'
    'PINN+RAR 1D         │ 8.37  │ 166с   │ Нет     │ ~ (RAR не помог)\n'
    'DeepONet 1D         │ 0.62  │ 310с   │ Да ★    │ В 13× точнее PINN\n'
    'PINN 2D             │ 0.53  │ 202с   │ Нет     │ Гладкое решение\n'
    'DeepONet 2D MINI    │ 0.28  │ 69с    │ Да ★★   │ Абсолютный лидер!\n\n'
    '★  Лучший в 1D: DeepONet (L2=0.62, но нужны данные)\n'
    '★★ Лучший в 2D: DeepONet MINI (L2=0.28, быстрее всех!)',
    top=1.2, left=0.3, width=7, height=6, font_size=14)

add_image_safe(slide, os.path.join(BASE, 'comparison_metrics_bar.png'),
               left=7.5, top=1.2, width=5.5, height=5.8)
add_slide_number(slide, 21)

# ============================================================
# СЛАЙД 22: Сравнение Loss-кривых
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Сравнение Loss-кривых всех методов')
add_image_safe(slide, os.path.join(BASE, 'comparison_loss_all.png'),
               left=0.5, top=1.3, width=12.0, height=5.8)
add_slide_number(slide, 22)

# ============================================================
# СЛАЙД 23: Анализ сходимости DeepONet
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Анализ сходимости DeepONet')
add_body(slide,
    '1D DeepONet:\n'
    '• Быстрое плато: сеть выучивает оператор за ~2000 эпох\n'
    '• Дальнейшее обучение почти не улучшает результат\n'
    '• Причина: низкоразмерная зависимость u₀ → u(x,t)\n\n'
    '2D DeepONet MINI:\n'
    '• Плавное падение loss в 13 раз (5.5×10⁻² → 4.1×10⁻³)\n'
    '• Стабильная сходимость без осцилляций и взрывов градиента\n'
    '• Причина: малая размерность (PCA) + 600 примеров\n\n'
    '2D DeepONet (полный, 400 примеров):\n'
    '• Loss застыл на 0.12 за 30000 эпох\n'
    '• Проклятие размерности: 1024D вход × 400 примеров\n'
    '• Вывод: DeepONet критически зависит от качества датасета',
    font_size=14)
add_slide_number(slide, 23)

# ============================================================
# СЛАЙД 24: Обсуждение
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Обсуждение результатов')
add_body(slide,
    'ЧТО СРАБОТАЛО ХОРОШО:\n'
    '✓ PINN успешно решает уравнение Бюргерса без данных\n'
    '✓ DeepONet в 1D даёт отличную точность (L2=0.62)\n'
    '✓ DeepONet 2D MINI показал лучший результат (L2=0.28)\n'
    '✓ 2D задача проще для PINN (гладкое решение, нет острых фронтов)\n\n'
    'ЧТО МОЖНО УЛУЧШИТЬ:\n'
    '• PINN 1D: увеличить число точек у фронта\n'
    '• RAR: применить при меньшей вязкости (ν < 0.001)\n'
    '• DeepONet 2D: больше данных или FNO-архитектура\n\n'
    'ОГРАНИЧЕНИЯ:\n'
    '• Всё обучение на CPU (Windows, нет GPU)\n'
    '• Ограниченное время на подбор гиперпараметров',
    font_size=13)
add_slide_number(slide, 24)

# ============================================================
# СЛАЙД 25: Выводы
# ============================================================
slide = add_slide(prs)
add_title(slide, 'Выводы')
add_body(slide,
    '1. PINN — универсальный инструмент для решения PDE:\n'
    '   • Не требует данных — только уравнение и граничные условия\n'
    '   • Хорош для гладких решений (2D: L2=0.53)\n'
    '   • Требует настройки распределения коллокационных точек\n\n'
    '2. RAR-адаптация — не всегда эффективна:\n'
    '   • Даёт выигрыш только при острых фронтах (малая вязкость)\n'
    '   • При ν=0.003 результат сопоставим с обычным PINN\n\n'
    '3. DeepONet — лучший выбор для суррогатного моделирования:\n'
    '   • 1D: L2=0.62 — в 13 раз точнее PINN\n'
    '   • 2D MINI: L2=0.28 — абсолютный рекорд среди всех моделей!\n'
    '   • После обучения — мгновенные предсказания (доли секунды)\n\n'
    '4. Перспективы развития:\n'
    '   • Применение Fourier Neural Operator (FNO) для 2D задач\n'
    '   • Обобщение на уравнения Навье-Стокса\n'
    '   • Использование GPU для ускорения обучения',
    font_size=12)
add_slide_number(slide, 25)

# ============================================================
# СОХРАНЕНИЕ
# ============================================================
output_path = os.path.join(BASE, 'Burgers_PINN_DeepONet_presentation.pptx')
prs.save(output_path)
print(f"\n{'='*60}")
print(f"Presentation saved: {output_path}")
print(f"Total slides: {len(prs.slides)}")
print(f"{'='*60}")