import os
import torch

import os
import json

def generate_model_configs(output_dir, depths, base_widths, width_patterns):
    os.makedirs(output_dir, exist_ok=True)
    
    def compute_width(base_width, i, pattern):
        if pattern == 'constant':
            return base_width
        elif pattern == 'expand':
            return base_width * (2 ** i)
        elif pattern == 'shrink':
            return max(base_width // (2 ** i), 4)
        else:
            raise ValueError(f'Unknown pattern: {pattern}')
    
    for depth in depths:
        for base_width in base_widths:
            for pattern in width_patterns:
                layers = []
                for i in range(depth):
                    width = compute_width(base_width, i, pattern)
                    layers.append({
                        "type": "Linear",
                        "out_features": width,
                        "activation": "ReLU"
                    })
                layers.append({"type": "Linear", "out_features": 10})  # output layer
                
                config = {
                    "input_dim": None,  # to be filled dynamically
                    "layers": layers
                }
                
                filename = f"{pattern}_d{depth}_w{base_width}.json"
                with open(os.path.join(output_dir, filename), 'w') as f:
                    json.dump(config, f, indent=2)



def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
        path: str, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        epoch: int, 
        best_test_loss: float, 
        best_test_acc: float
    ):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_test_loss': best_test_loss,
        'best_test_acc': best_test_acc
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def load_model(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict['epoch'], state_dict['best_test_loss'], state_dict['best_test_acc']