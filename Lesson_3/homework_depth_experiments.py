from datasets import get_cifar_loaders, get_mnist_loaders
from utils.experiment_utils import train_model
from utils.visualization_utils import plot_learning_curves, save_results_table
from models import create_model_from_config
import os

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
    
    data = train_model(model, train_loader, test_loader, device='mps', epochs=25)
    
    return data
    

def main():
    models_configs_path = 'Lesson_3/experiment_1.2'
    # models_configs_path = 'Lesson_3/test'
    name_datasets = ['cifar', 'mnist']
    results = {'cifar': [], 'mnist': []}
    for dataset_name in name_datasets:
        for model_config in os.listdir(models_configs_path):
            model_config_path = '/'.join([models_configs_path, model_config])
            result = run_experiment(dataset_name, model_config_path)
            result['model_type'] = model_config.replace('.json', '')
            results[dataset_name].append(result)
    for dataset_name in name_datasets:
        save_results_table(results, dataset_name, save_dir='Lesson_3/results/depth_experiments/drop_norm')
  
    for dataset_name in name_datasets:
        plot_learning_curves(results[dataset_name], dataset_name, save_dir='Lesson_3/results/depth_experiments/drop_norm')


    
if __name__ == '__main__':
    main()