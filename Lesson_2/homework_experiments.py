import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from homework_datasets import CustomCSVDataset
from utils import train_model
from models.models import LogisticRegression

# Создаем папки для сохранения графиков, если их нет
os.makedirs("Lesson_2/plots/individual", exist_ok=True)
os.makedirs("Lesson_2/plots/summary", exist_ok=True)

# Загружаем датасет (Titanic) с указанием столбца с метками
dataset = CustomCSVDataset("Lesson_2/data/titanic.csv", label_column="survived")

# Задаем варианты гиперпараметров для перебора
learning_rates = [0.00001, 0.0001, 0.001]
batch_sizes = [16, 32, 64]
optimizers = ['SGD', 'Adam', 'RMSprop']

# Список для хранения результатов экспериментов
results = []

# Перебираем все комбинации learning_rate, batch_size и оптимизатора
for lr, bs, opt in itertools.product(learning_rates, batch_sizes, optimizers):
    print(f"--- Training: lr={lr}, batch_size={bs}, optimizer={opt} ---")

    # Создаем новую модель перед каждым обучением, чтобы не использовать предыдущие веса
    model = LogisticRegression(in_features=dataset.X.shape[1], num_classes=1)

    # Запускаем обучение модели с текущими гиперпараметрами
    train_data = train_model(model, dataset,
                             batch_size=bs,
                             lr=lr,
                             optimizer_name=opt,
                             epochs=50,
                             val_split=0.2,
                             verbose=False)

    # Получаем финальную точность на валидации
    acc = train_data.get('val_accuracy', 0)

    # Сохраняем результаты для анализа
    results.append({
        'lr': lr,
        'batch_size': bs,
        'optimizer': opt,
        'accuracy': acc
    })

    # Визуализируем процесс обучения: графики лосса и точности по эпохам
    plt.figure(figsize=(10, 4))

    # График функции потерь (train и val)
    plt.subplot(1, 2, 1)
    plt.plot(train_data['history']['train_loss'], label='Train Loss')
    plt.plot(train_data['history']['val_loss'], label='Val Loss')
    plt.title(f"Loss: lr={lr}, bs={bs}, opt={opt}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # График точности на валидации
    plt.subplot(1, 2, 2)
    plt.plot(train_data['history']['val_accuracy'], label='Val Accuracy', color='green')
    plt.title(f"Accuracy: lr={lr}, bs={bs}, opt={opt}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Формируем имя файла, меняем точки на 'p', чтобы избежать проблем с файловой системой
    filename = f"Lesson_2/plots/individual/lr{lr}_bs{bs}_opt{opt}.png".replace('.', 'p')
    # Сохраняем график
    plt.savefig(filename)
    plt.close()

# Конвертируем результаты в DataFrame для удобного анализа и визуализации
df_results = pd.DataFrame(results)

# Строим общий график влияния гиперпараметров на точность
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_results,
             x='lr', y='accuracy',
             hue='optimizer',
             style='batch_size',
             markers=True, dashes=False)
plt.title("Влияние learning rate, optimizer и batch size на точность")
plt.xscale("log")  # Логарифмическая шкала по learning rate для лучшей визуализации
plt.grid(True)

# Сохраняем общий график
summary_plot_path = "Lesson_2/plots/summary/hyperparameter_accuracy_plot.png"
plt.savefig(summary_plot_path)
plt.show()

# Находим лучшую конфигурацию гиперпараметров для каждого оптимизатора
best_per_optimizer = df_results.groupby("optimizer").apply(
    lambda df: df.sort_values("accuracy", ascending=False).iloc[0]
).reset_index(drop=True)

print("Лучшие результаты по оптимизаторам:")
print(best_per_optimizer)

# Сохраняем таблицу с лучшими результатами
best_per_optimizer.to_csv("Lesson_2/plots/summary/best_results.csv", index=False)
