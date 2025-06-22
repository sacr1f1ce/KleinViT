import os
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model.vit import ViT
from src.klein_tools.feature_extractor import KleinFeatureExtractor

# --- Configuration ---
MODELS_TO_EVALUATE = {
    'Baseline': {
        'path': './models/vit_baseline.pth',
        'params': {
            'use_klein_features': False,
            'use_topological_attention': False,
            'use_gated_attention': False
        }
    },
    'Approach A (Concat)': {
        'path': './models/vit_approach_a.pth',
        'params': {
            'use_klein_features': True,
            'use_topological_attention': False,
            'use_gated_attention': False
        }
    },
    'Approach B (Bias)': {
        'path': './models/vit_approach_b.pth',
        'params': {
            'use_klein_features': False,
            'use_topological_attention': True,
            'use_gated_attention': False
        }
    },
    'Approach C (Gated)': {
        'path': './models/vit_approach_c.pth',
        'params': {
            'use_klein_features': False,
            'use_topological_attention': False,
            'use_gated_attention': True
        }
    }
}
OUTPUT_DIR = './output'
DEFAULT_MODEL_PARAMS = {
    'image_size': 33, 'patch_size': 3, 'num_classes': 10,
    'dim': 256, 'depth': 6, 'heads': 8, 'mlp_dim': 512,
}

# --- Plotting Functions ---


def plot_curves(all_logs, metric, plot_type, title, filename, log_scale=False):
    """
    Generates and saves plots for training and testing metrics.

    Args:
        all_logs (dict): Dictionary of pandas DataFrames with log data.
        metric (str): The metric to plot (e.g., 'loss', 'acc').
        plot_type (str): 'train' or 'test'.
        title (str): The title for the plot.
        filename (str): The filename to save the plot to.
        log_scale (bool): Whether to use a logarithmic scale for the y-axis.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    for name, log_df in all_logs.items():
        epochs = log_df.index
        mean = log_df[f'{plot_type}_{metric}_mean']
        std = log_df[f'{plot_type}_{metric}_std']

        ax.plot(epochs, mean, label=f'{name}')
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    if log_scale:
        ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close(fig)

# --- Evaluation Function ---


def evaluate_model(model, device, testloader, klein_feature_extractor=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc='Final Evaluation')
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            use_features = getattr(model, 'use_klein_features', False) or \
                getattr(model, 'use_topological_attention', False) or \
                getattr(model, 'use_gated_attention', False)

            if use_features:
                if klein_feature_extractor is None:
                    raise ValueError(
                        "Model requires Klein features, but no extractor was provided."
                    )
                klein_features = klein_feature_extractor.get_klein_coordinates(
                    inputs)
                outputs = model(inputs, klein_features=klein_features)
            else:
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# --- Main ---


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Instantiate the feature extractor
    # It will be used for any model that needs it.
    # The patch size must match the model's patch size.
    klein_feature_extractor = KleinFeatureExtractor(
        patch_size=DEFAULT_MODEL_PARAMS['patch_size'],
        device=device
    )

    # Load test data
    transform_test = transforms.Compose([
        transforms.Pad((0, 0, 1, 1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2)

    all_logs = {}
    final_accuracies = {}

    print("\n--- Loading Logs and Evaluating Models ---")
    for name, config in MODELS_TO_EVALUATE.items():
        print(f"\nProcessing Model: {name}")

        # Load logs
        log_path = config['path'].replace('.pth', '_logs.csv')
        if os.path.exists(log_path):
            all_logs[name] = pd.read_csv(log_path, index_col='epoch')
            print(f"  - Logs loaded from {log_path}")
        else:
            print(f"  - WARNING: Log file not found at {log_path}")
            continue

        # Load model and evaluate
        if os.path.exists(config['path']):
            model_params = {**DEFAULT_MODEL_PARAMS, **config['params']}
            model = ViT(**model_params).to(device)

            checkpoint = torch.load(config['path'], map_location=device)
            model.load_state_dict(checkpoint['model'])
            print(
                f"  - Model loaded from {config['path']} "
                f"(Epoch {checkpoint.get('epoch', 'N/A')})"
            )

            final_acc = evaluate_model(
                model, device, testloader, klein_feature_extractor)
            final_accuracies[name] = final_acc
            print(f"  - Final evaluated accuracy: {final_acc:.2f}%")
        else:
            print(f"  - WARNING: Model file not found at {config['path']}")
            final_accuracies[name] = 0

    # --- Generate Plots ---
    if all_logs:
        print("\n--- Generating Plots ---")
        # Plot Loss Curves
        plot_curves(all_logs, 'loss', 'train',
                    'Training Loss Curves (Log Scale)',
                    'train_loss_curves.png', log_scale=True)
        plot_curves(all_logs, 'loss', 'test',
                    'Test Loss Curves (Log Scale)',
                    'test_loss_curves.png', log_scale=True)

        # Prepare logs for accuracy plotting
        renamed_logs = {}
        for name, df in all_logs.items():
            renamed_df = df.rename(
                columns={'train_acc': 'train_acc_mean',
                         'test_acc': 'test_acc_mean'}
            )
            renamed_df['train_acc_std'] = 0  # No std for total acc
            renamed_df['test_acc_std'] = 0
            renamed_logs[name] = renamed_df

        # Plot Accuracy Curves
        plot_curves(renamed_logs, 'acc', 'train',
                    'Training Accuracy Curves', 'train_accuracy_curves.png')
        plot_curves(renamed_logs, 'acc', 'test',
                    'Test Accuracy Curves', 'test_accuracy_curves.png')

    # --- Print Final Report ---
    print("\n\n--- Final Performance Report ---")
    report_df = pd.DataFrame.from_dict(
        final_accuracies, orient='index', columns=['Final Test Accuracy'])
    report_df = report_df.sort_values(
        by='Final Test Accuracy', ascending=False)
    report_df['Final Test Accuracy'] = report_df['Final Test Accuracy'].map(
        '{:.2f}%'.format)
    print(report_df.to_markdown())


if __name__ == '__main__':
    main()
