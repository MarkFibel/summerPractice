import os
import matplotlib.pyplot as plt
from models import create_model_from_config
from torch import optim
from utils.experiment_utils import train_model
import torch.nn as nn
import torch
import json
import pandas as pd
from datasets import get_cifar_loaders



def run_all_experiments(config_dir, train_loader, test_loader, input_size, num_classes, device='cpu', output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    config_files = sorted([f for f in os.listdir(config_dir) if f.endswith('.json')])
    
    for config_file in config_files:
        name = config_file.replace('.json', '')
        print(f'\n🔍 Запуск конфигурации: {name}')
        
        config_path = os.path.join(config_dir, config_file)
        model = create_model_from_config(config_path, input_size=input_size, num_classes=num_classes).to(device)

        # Настройка оптимизатора
        if "l2" in name:
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        result = train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device=device, optimizer=optimizer)
        results[name] = result

        # Сохраняем модель
        torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_model.pt"))
    
    return results


def plot_accuracy_curves(results, output_dir):
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(res['test_accs'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('📈 Точность на тесте при разных регуляризациях')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'))
    plt.close()


def plot_weight_distributions(results, output_dir):
    for name, res in results.items():
        model = res['model']
        weights = []
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                weights.extend(layer.weight.detach().cpu().numpy().flatten())
        
        plt.figure(figsize=(6, 4))
        plt.hist(weights, bins=100)
        plt.title(f'Распределение весов: {name}')
        plt.xlabel('Значение веса')
        plt.ylabel('Частота')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_weights.png'))
        plt.close()


def save_summary_table(results, output_dir):
    summary = []
    for name, res in results.items():
        final_acc = res['test_accs'][-1] * 100
        train_time = res['train_time']
        summary.append({
            'Конфигурация': name,
            'Точность (%)': round(final_acc, 2),
            'Время (с)': round(train_time, 1)
        })

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    print(df.to_markdown(index=False))


# 🚀 Запуск
if __name__ == "__main__":
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = 'Lesson_3/results/regularization_experiments/exps'
    os.makedirs(output_dir, exist_ok=True)

    results = run_all_experiments(
        config_dir='Lesson_3/experiment_dop',
        train_loader=train_loader,
        test_loader=test_loader,
        input_size=32 * 32 * 3,
        num_classes=10,
        device=device,
        output_dir=output_dir
    )

    plot_accuracy_curves(results, output_dir)
    plot_weight_distributions(results, output_dir)
    save_summary_table(results, output_dir)
