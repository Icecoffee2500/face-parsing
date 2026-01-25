import torch
import numpy as np
import random

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Argument Parser for Training Configuration')

    # General
    parser.add_argument('--device-id', type=int, default=0, help='Device ID')

    # Dataset
    parser.add_argument('--num-classes', type=int, default=19, help='Number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=12, help='Number of workers for data loading')
    # parser.add_argument('--image-size', type=int, nargs=2, default=[448, 448], help='Size of input images')
    parser.add_argument('--label-size', type=int, nargs=2, default=[512, 512], help='Size of input labels')
    parser.add_argument(
        '--data-root', type=str, default='dataset/CelebAMask-HQ/', help='Root directory of the dataset'
    )
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument(
        '--split-file', type=str, default='dataset/CelebAMask-HQ/train_val_split.json', help='Path to saved split file'
    )

    # Optimizer
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--lr-start', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--max-iter', type=int, default=80000, help='Maximum number of iterations')
    parser.add_argument('--power', type=float, default=0.9, help='Power for learning rate policy')
    parser.add_argument('--lr-warmup-epochs', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--warmup-start-lr', type=float, default=1e-5, help='Warmup starting learning rate')
    parser.add_argument('--score-thres', type=float, default=0.7, help='Score threshold')
    parser.add_argument(
        '--ignore-background',
        action='store_true',
        help='Ignore background (class 0) in loss and metrics',
    )

    # Training loop
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone architecture')

    # Train loop
    parser.add_argument('--print-freq', type=int, default=100, help='Print frequency during training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='face-parsing', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity/team (optional)')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name (optional)')
    parser.add_argument('--wandb-log-images', action='store_true', help='Log sample images to W&B')
    parser.add_argument('--wandb-image-count', type=int, default=4, help='Number of images to log per eval')

    args = parser.parse_args()
    return args


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)