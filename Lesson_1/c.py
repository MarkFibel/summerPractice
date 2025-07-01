import torch

# 1.1
rand_tensor = torch.rand((3, 4))
zeros_tensor = torch.zeros((2, 3, 4))
ones_tensor = torch.ones((5, 5))
randint_tensor = torch.arange(16).reshape(4, 4)

print("3x4 случайный тензор:\n", rand_tensor)
print("2x3x4 нулевой тензор:\n", zeros_tensor)
print("5x5 единичный тензор:\n", ones_tensor)
print("4x4 тензор с числами от 0 до 15:\n", randint_tensor)

# 1.2
A = torch.randn((3, 4))
B = torch.randn((4, 3))

print(f'A = {A}')
print(f'B = {B}')

A_T = A.T
matmul_result = torch.mm(A, B)
B_T = B.t()
elementwise_mul = A * B_T
Asum = A.sum()

print("Транспонированный A:\n", A_T)
print("Матричное умножение A и B:\n", matmul_result)
print("Поэлементное умножение A и транспонированного B:\n", elementwise_mul)
print("Сумма всех элементов A:", Asum)

# 1.3
tensor = torch.randn((5, 5, 5))

first_row = tensor[0, :, :]
last_column = tensor[:, :, -1]
center_submatrix = tensor[2:4, 2:4, 2:4]
even_indices_elements = tensor[::2, ::2, ::2]

print("Первая строка:\n", first_row)
print("Последний столбец:\n", last_column)
print("Центральная 2x2x2 подматрица:\n", center_submatrix)
print("Элементы с четными индексами:\n", even_indices_elements)

tensor = torch.randn((24))

tensor_2x12 = tensor.reshape(2, 12)
tensor_3x8 = tensor.reshape(3, 8)
tensor_4x6 = tensor.reshape(4, 6)
tensor_2x3x4 = tensor.reshape(2, 3, 4)
tensor_2x2x2x3 = tensor.reshape(2, 2, 2, 3)

print("2x12:\n", tensor_2x12)
print("3x8:\n", tensor_3x8)
print("4x6:\n", tensor_4x6)
print("2x3x4:\n", tensor_2x3x4)
print("2x2x2x3:\n", tensor_2x2x2x3)