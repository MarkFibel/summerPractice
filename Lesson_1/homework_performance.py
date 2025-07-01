import torch
import time

# 3.1
tensor_64x1024x1024 = torch.randn((64, 1024, 1024))
tensor_128x512x512 = torch.randn((128, 512, 512))
tensor_256x256x256 = torch.randn((256, 256, 256))

# 3.2
def measure_time_cpu(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    # print(f"CPU время выполнения: {end_time - start_time:.6f} секунд")
    return result, end_time - start_time

def measure_time_mps(func, *args, **kwargs):
    device = torch.device('mps')  # Указываем устройство MPS
    # Передача данных на устройство MPS
    args = [arg.to(device) if hasattr(arg, 'to') else arg for arg in args]
    kwargs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in kwargs.items()}

    # Засекаем время до выполнения
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    # Засекаем время после выполнения
    end_time = time.perf_counter()

    elapsed_time_ms = (end_time - start_time)
    # Вернем результат и время
    return result, elapsed_time_ms

# 3.3
mps_device = torch.device('mps')
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

def matmul_cpu(lst):
    times = []
    for mat in lst:
        result, time_cpu = measure_time_cpu(torch.matmul, mat, mat)
        times.append(time_cpu)

    return times

def matmul_mps(lst):
    times = []
    for mat in lst:
        result, time_mps = measure_time_mps(torch.matmul, mat, mat)
        times.append(time_mps)

    return times

def add_cpu(lst):
    times = []
    for mat in lst:
        result, time_cpu = measure_time_cpu(torch.add, mat, mat)
        times.append(time_cpu)

    return times

def add_mps(lst):
    times = []
    for mat in lst:
        result, time_mps = measure_time_mps(torch.add, mat, mat)
        times.append(time_mps)

    return times

def mul_cpu(lst):
    times = []
    for mat in lst:
        result, time_cpu = measure_time_cpu(torch.mul, mat, mat)
        times.append(time_cpu)

    return times

def mul_mps(lst):
    times = []
    for mat in lst:
        result, time_mps = measure_time_mps(torch.mul, mat, mat)
        times.append(time_mps)

    return times

def sum_cpu(lst):
    times = []
    for mat in lst:
        result, time_cpu = measure_time_cpu(torch.sum, mat)
        times.append(time_cpu)

    return times

def sum_mps(lst):
    times = []
    for mat in lst:
        result, time_mps = measure_time_mps(torch.sum, mat)
        times.append(time_mps)

    return times

def T_cpu(lst):
    times = []
    for mat in lst:
        start_time = time.time()
        result = mat.T
        end_time = time.time()
        time_cpu = end_time - start_time
        times.append(time_cpu)

    return times

def T_mps(lst):
    times = []
    for mat in lst:
        mat = mat.to(torch.device('mps'))
        start_time = time.time()
        result = mat.T
        end_time = time.time()
        time_mps = end_time - start_time
        times.append(time_mps)

    return times

matmul_cpu_times = matmul_cpu([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])
matmul_mps_times = matmul_mps([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])

add_cpu_times = add_cpu([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])
add_mps_times = add_mps([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])

mul_cpu_times = mul_cpu([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])
mul_mps_times = mul_mps([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])

T_cpu_times = T_cpu([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])
T_mps_times = T_mps([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])

sum_cpu_times = sum_cpu([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])
sum_mps_times = sum_mps([tensor_64x1024x1024, tensor_128x512x512, tensor_256x256x256])


mps_times = [matmul_mps_times, add_mps_times, mul_mps_times, T_mps_times, sum_mps_times]
cpu_times = [matmul_cpu_times, add_cpu_times, mul_cpu_times, T_cpu_times, sum_cpu_times]

func_names = ['Матричное умножение', 'Поэлементное сложение', 'Поэлементное умножение', 'Транспонирование', 'Вычисление суммы всех элементов']
tensors = ['64x1024x1024', '128x512x512', '256x256x256']
speedup = []
for mps_time, cpu_time in zip(mps_times, cpu_times):
    coef = sum(cpu_time) / sum(mps_time) / len(tensors)
    speedup.append(coef)

import pandas as pd

def build_results_table(func_names, tensors, cpu_times, mps_times):
    # Создаем список для строк таблицы
    table_rows = []

    # Заполняем строки для каждого метода и каждого тензора
    for i, name in enumerate(func_names):
        for j, tensor_size in enumerate(tensors):
            row = {
                'Название операции': name,
                'Размер тензора': tensor_size,
                'Время CPU (сек)': f"{cpu_times[i][j]:.4f}",
                'Время MPS (сек)': f"{mps_times[i][j]:.4f}",
                'Ускорение': f"{cpu_times[i][j] / mps_times[i][j]:.2f}" if mps_times[i][j] != 0 else 'inf'
            }
            table_rows.append(row)

    # Создаем DataFrame из списка строк
    df = pd.DataFrame(table_rows)

    # Вычисляем средние по каждому методу
    avg_rows = []
    for i, name in enumerate(func_names):
        cpu_avg = sum(cpu_times[i]) / len(cpu_times[i])
        mps_avg = sum(mps_times[i]) / len(mps_times[i])
        speedup_avg = cpu_avg / mps_avg if mps_avg != 0 else float('inf')
        avg_row = {
            'Название операции': f'{name} (среднее)',
            'Размер тензора': '-',
            'Время CPU (сек)': f"{cpu_avg:.4f}",
            'Время MPS (сек)': f"{mps_avg:.4f}",
            'Ускорение': f"{speedup_avg:.2f}"
        }
        avg_rows.append(avg_row)

    # Добавляем средние строки
    df = pd.concat([df, pd.DataFrame(avg_rows)], ignore_index=True)

    return df

# Вызов функции для построения таблицы
results_df = build_results_table(func_names, tensors, cpu_times, mps_times)

# Сохраняем таблицу в markdown файл
with open('results_table.md', 'w', encoding='utf-8') as f:
    f.write(results_df.to_markdown(index=False))