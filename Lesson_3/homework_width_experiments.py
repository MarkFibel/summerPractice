from datasets import get_cifar_loaders, get_mnist_loaders
from utils.experiment_utils import train_model
from utils.visualization_utils import plot_learning_curves, save_results_table, summarize_training_runs_to_png, plot_heatmaps
from utils.model_utils import count_parameters, generate_model_configs
from models import create_model_from_config
import os, json

def run_experiment(dataset_name: str, config_path: str):
    if dataset_name == 'cifar':
        train_loader, test_loader = get_cifar_loaders()
        input_size = 3072
    elif dataset_name == 'mnist':
        train_loader, test_loader = get_mnist_loaders()
        input_size = 784
    else:
        raise ValueError('Unknown dataset name')
    
    model = create_model_from_config(config_path, input_size=input_size).to('mps')
    
    data = train_model(model, train_loader, test_loader, device='mps', epochs=15)
    
    return data

def run_all_experiments(models_configs_path, dataset_names):
    results = {name: [] for name in dataset_names}

    for dataset_name in dataset_names:
        if dataset_name == 'cifar':
            train_loader, test_loader = get_cifar_loaders()
            input_size = 32 * 32 * 3
        elif dataset_name == 'mnist':
            train_loader, test_loader = get_mnist_loaders()
            input_size = 28 * 28
        else:
            raise ValueError("Unsupported dataset")

        for config_file in os.listdir(models_configs_path):
            config_path = os.path.join(models_configs_path, config_file)

            # Только теперь читаем config для анализа, но не передаём в модель
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if config['input_dim'] is None:
                if dataset_name == 'cifar':
                    config['input_dim'] = 32 * 32 * 3
                elif dataset_name == 'mnist':
                    config['input_dim'] = 28 * 28
                # Сохраняем обратно — если хочешь, чтобы модель прочитала актуальный input_dim
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

            model = create_model_from_config(config_path, input_size).to('mps')

            result = train_model(model, train_loader, test_loader, device='mps', epochs=15)
            result['model_type'] = config_file.replace('.json', '')
            result['params'] = count_parameters(result['model'])
            results[dataset_name].append(result)

    return results

    

def main():
    config_dir = 'Lesson_3/experiment_2.2'
    results_dir = 'Lesson_3/results/width_experiments/best_arc'
    dataset_names = ['cifar']  # или ['mnist', 'cifar']

    generate_model_configs(
        output_dir=config_dir,
        depths=[2, 3, 4, 5, 6],
        base_widths=[16, 32, 64],
        width_patterns=['constant', 'expand', 'shrink']
    )

    results = run_all_experiments(config_dir, dataset_names)

    from utils.visualization_utils import (
        plot_learning_curves,
        summarize_training_runs_to_png,
        save_results_table
    )

    for dataset_name in dataset_names:
        summarize_training_runs_to_png(results[dataset_name], filename=f'{results_dir}/summary_{dataset_name}.png')
        save_results_table(results, dataset_name, save_dir=results_dir)
        plot_learning_curves(results[dataset_name], dataset_name, save_dir=results_dir)
        plot_heatmaps(results, dataset_name, save_dir=results_dir)


    
if __name__ == '__main__':
    main()