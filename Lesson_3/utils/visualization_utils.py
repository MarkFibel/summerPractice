import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def plot_heatmaps(results, dataset_name, save_dir):
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame([
        {
            'depth': int(r['model_type'].split('_d')[1].split('_')[0]),
            'width': int(r['model_type'].split('_w')[1]),
            'pattern': r['model_type'].split('_')[0],
            'test_acc': r['test_accs'][-1]
        }
        for r in results[dataset_name]
    ])
    
    for pattern in df['pattern'].unique():
        df_pattern = df[df['pattern'] == pattern].copy()
        df_pattern = df_pattern.sort_values(by=['depth', 'width'])
        pivot = df_pattern.pivot(index='depth', columns='width', values='test_acc')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f'{dataset_name.upper()} - Pattern: {pattern}')
        plt.savefig(os.path.join(save_dir, f'heatmap_{dataset_name}_{pattern}.png'))
        plt.close()




def summarize_training_runs_to_png(results_list, filename="model_summary.png"):
    """
    results_list: список словарей с полями:
        - model_type
        - params
        - train_time
    filename: путь для сохранения PNG
    """
    # Преобразуем в DataFrame
    df = pd.DataFrame(results_list)
    df = df.rename(columns={
        "model_type": "Модель",
        "params": "Параметры",
        "train_time": "Время обучения (сек)"
    })
    
    # Формат чисел
    df["Параметры"] = df["Параметры"].apply(lambda x: f"{x:,}")
    df["Время обучения (сек)"] = df["Время обучения (сек)"].apply(lambda x: f"{x:.2f}")
    df = df[["Модель", "Параметры", "Время обучения (сек)"]]
    # Построим таблицу
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.5 * len(df)))  # динамический размер
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Таблица сохранена как: {filename}")
    

def save_results_table(results, dataset_name, save_dir='tables'):
    """
    Сохраняет таблицу результатов по точности моделей в виде PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Заголовок таблицы
    col_labels = ["Model", "Train Accuracy", "Test Accuracy"]
    cell_text = []

    for result in results[dataset_name]:
        row = [
            result['model_type'],
            f"{result['train_accs'][-1]:.4f}",
            f"{result['test_accs'][-1]:.4f}"
        ]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(8, len(cell_text) * 0.6 + 1))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title(f'Results for {dataset_name.upper()}', fontsize=14, weight='bold', pad=20)
    save_path = os.path.join(save_dir, f'{dataset_name}_results_table.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'Таблица результатов сохранена в {save_path}')



def plot_training_history(history, save_path=None):
    """Визуализирует историю обучения. Если указан save_path — сохраняет график."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"График сохранён в {save_path}")
        plt.close()
    else:
        plt.show()


def plot_learning_curves(results, dataset_name, save_dir=None):
    """
    Визуализирует кривые обучения для всех моделей.
    Если указан save_dir, сохраняет каждый график в этот каталог.
    """
    for result in results:
        label = result['model_type']
        epochs = range(1, len(result['train_losses']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, result['train_losses'], label='Train')
        plt.plot(epochs, result['test_losses'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{label} - Loss on {dataset_name}')
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, result['train_accs'], label='Train')
        plt.plot(epochs, result['test_accs'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{label} - Accuracy on {dataset_name}')
        plt.legend()

        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{label}_{dataset_name}.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            print(f"Сохранён график: {save_path}")
            plt.close()
        else:
            plt.show()
