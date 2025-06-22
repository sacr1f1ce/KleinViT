import argparse
import os
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from src.model.vit import ViT
from src.klein_tools.feature_extractor import KleinFeatureExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Train a Vision Transformer on CIFAR-10.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training.')
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Patch size for ViT.')
    parser.add_argument('--dim', type=int, default=256,
                        help='Embedding dimension.')
    parser.add_argument('--depth', type=int, default=6,
                        help='Depth of the transformer.')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument('--mlp_dim', type=int, default=512,
                        help='MLP dimension in transformer.')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset.')
    parser.add_argument('--save_path', type=str, default='./models/vit_cifar10.pth',
                        help='Path to save trained models.')
    parser.add_argument('--use_klein_features', action='store_true',
                        help='Use Klein bottle topological features (Approach A).')
    parser.add_argument('--use_topological_attention', action='store_true',
                        help='Use topological attention (Approach B).')
    parser.add_argument('--use_gated_attention', action='store_true',
                        help='Use gated topological attention (Approach C).')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Pad((0, 0, 1, 1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Pad((0, 0, 1, 1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    model = ViT(
        image_size=33,
        patch_size=args.patch_size,
        num_classes=10,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=0.1,
        emb_dropout=0.1,
        use_klein_features=args.use_klein_features,
        use_topological_attention=args.use_topological_attention,
        use_gated_attention=args.use_gated_attention
    ).to(device)

    feature_extractor = None
    if args.use_klein_features or args.use_topological_attention or args.use_gated_attention:
        print("==> Initializing Klein Feature Extractor...")
        feature_extractor = KleinFeatureExtractor(
            patch_size=args.patch_size,
            device=device
        )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_acc = 0
    history = {
        'train_loss_mean': [], 'train_loss_std': [], 'train_acc': [],
        'test_loss_mean': [], 'test_loss_std': [], 'test_acc': [],
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch+1}/{args.epochs}")

        # Train
        model.train()
        batch_losses = []
        correct, total = 0, 0
        progress_bar = tqdm(trainloader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            klein_coords = None
            if args.use_klein_features or args.use_topological_attention or args.use_gated_attention:
                with torch.no_grad():
                    klein_coords = feature_extractor.get_klein_coordinates(
                        inputs)

            outputs = model(inputs, klein_features=klein_coords)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix(
                loss=np.mean(batch_losses), acc=100.*correct/total)

        history['train_loss_mean'].append(np.mean(batch_losses))
        history['train_loss_std'].append(np.std(batch_losses))
        history['train_acc'].append(100.*correct/total)

        # Test
        model.eval()
        batch_losses = []
        correct, total = 0, 0
        with torch.no_grad():
            progress_bar = tqdm(testloader, desc='Testing')
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device), targets.to(device)

                klein_coords = None
                if args.use_klein_features or args.use_topological_attention or args.use_gated_attention:
                    klein_coords = feature_extractor.get_klein_coordinates(
                        inputs)

                outputs = model(inputs, klein_features=klein_coords)
                loss = criterion(outputs, targets)
                batch_losses.append(loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar.set_postfix(
                    loss=np.mean(batch_losses), acc=100.*correct/total)

        history['test_loss_mean'].append(np.mean(batch_losses))
        history['test_loss_std'].append(np.std(batch_losses))
        history['test_acc'].append(100.*correct/total)

        # Save checkpoint
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving checkpoint..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(args.save_path)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(state, args.save_path)
            best_acc = acc

        scheduler.step()

    print(f"Best Accuracy: {best_acc}%")

    # Save Training History
    log_path = args.save_path.replace('.pth', '_logs.csv')
    print(f"Saving training history to {log_path}")
    pd.DataFrame(history).to_csv(log_path, index_label='epoch')


if __name__ == '__main__':
    main()
