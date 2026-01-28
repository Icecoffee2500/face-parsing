import os
import argparse
import torch
from torch.utils.data import DataLoader

from models.bisenet import BiSeNet
from utils.celebamask_hq import CelebAMaskHQ
from utils.loss import OhemLossWrapper
from utils.function import evaluate
from utils.utils import random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Test/evaluate a trained model')
    parser.add_argument('--data-root', type=str, default='dataset/CelebAMask-HQ/', help='Dataset root')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pt)')
    parser.add_argument('--resolution', type=int, default=512, help='Test resolution (e.g., 512)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--num-classes', type=int, default=19, help='Number of classes')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone architecture')
    parser.add_argument('--device-id', type=int, default=0, help='CUDA device id')
    parser.add_argument('--ignore-background', action='store_true', help='Ignore background for metrics')
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--vis-dir', type=str, default='visualization/test', help='Visualization output dir')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main(params):
    random_seed(params.seed)
    device = torch.device(f'cuda:{params.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    weights_name = os.path.basename(params.weights)
    if params.backbone == 'resnet18':
        if 'resnet34' in weights_name:
            params.backbone = 'resnet34'
        elif 'resnet50' in weights_name:
            params.backbone = 'resnet50'

    val_dataset = CelebAMaskHQ(
        params.data_root,
        'val',
        resolution=[params.resolution, params.resolution],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = BiSeNet(num_classes=params.num_classes, backbone_name=params.backbone)
    state = torch.load(params.weights, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f'Failed to load weights. Check backbone with --backbone. Error: {exc}'
        ) from exc
    model.to(device)

    ignore_index = 0 if params.ignore_background else None
    n_min = params.batch_size * params.resolution * params.resolution // 16
    criterion = OhemLossWrapper(thresh=0.7, min_kept=n_min, ignore_index=ignore_index)

    class_names = [
        'background',
        'skin',
        'l_brow',
        'r_brow',
        'l_eye',
        'r_eye',
        'eye_g',
        'l_ear',
        'r_ear',
        'ear_r',
        'nose',
        'mouth',
        'u_lip',
        'l_lip',
        'neck',
        'neck_l',
        'cloth',
        'hair',
        'hat',
    ]

    metrics = evaluate(
        model,
        criterion,
        val_loader,
        device,
        params.num_classes,
        lb_ignore=0 if params.ignore_background else 255,
        class_names=class_names,
        print_freq=params.print_freq,
        log_images=False,
        log_step=None,
        epoch=0,
        vis_dir=params.vis_dir,
    )

    print(f"Test mIoU: {metrics['val/miou']:.6f}")
    print(f"Test macro F1: {metrics['val/macro_f1']:.6f}")
    print(f"Test micro F1: {metrics['val/micro_f1']:.6f}")
    print(f"Test per IoU: {metrics['val/per_iou']}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
