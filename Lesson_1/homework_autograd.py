import torch

x = torch.randn((1, ), requires_grad=True)
y = torch.randn((1, ), requires_grad=True)
z = torch.randn((1, ), requires_grad=True)

# Вычисляем функцию
f = x**2 + y**2 + z**2 + 2 * x * y * z

# Обратное распространение для получения градиентов
f.backward()

# Выводим градиенты
print("Градиенты:")
print("df/dx =", x.grad)
print("df/dy =", y.grad)
print("df/dz =", z.grad)

# df/dx = 2x + 2*y*z
# df/dy = 2y + 2*x*z
# df/dz = 2z + 2*x*y
print("\nГрадиенты аналитические:")
print("df/dx =", (2 * x + 2 * y * z).item())
print("df/dy =", (2 * y + 2 * x * z).item())
print("df/dz =", (2 * z + 2 * x * y).item())

# 2.2
x = torch.randint(0, 5, (5, ), requires_grad=False)
y_true = torch.randint(0, 3, (5, ), requires_grad=False)

w = torch.randn((1, ), requires_grad=True)
b = torch.zeros((1, ), requires_grad=True)

# Предсказания модели
y_pred = w * x + b

# Функция MSE
n = x.shape[0]
mse = (1/n) * torch.sum((y_pred - y_true) ** 2)

# Обратное распространение для получения градиентов
mse.backward()

print(f"\nГрадиенты:\ndMSE/dw = {w.grad}\ndMSE/db = {b.grad}")

# 2.3
def f(x):
    return torch.sin(x**2 + 1)

x = torch.randn((1, ), requires_grad=True)

y = f(x)

# df/dx = 2x * cos(x^2 + 1)

grad_x, = torch.autograd.grad(outputs=y, inputs=x)
hand_grad_x = 2 * x * torch.cos(x ** 2 + 1)

print(f"\nЗначение функции f(x): {y.item()}")
print(f"Градиент df/dx: {grad_x.item()}")
print(f"Градиент df/dx аналитически: {hand_grad_x.item()}")
