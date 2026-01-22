import os
from datetime import datetime
import importlib
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import PolynomialLR

from models.bisenet import BiSeNet
from utils.loss import OhemLossWrapper
from utils.transform import TrainTransform, DefaultTransform
from utils.dataset import CelebAMaskHQ, load_or_create_split
from utils.utils import parse_args, random_seed
from utils.function import train_one_epoch, evaluate, add_weight_decay

try:
    wandb = importlib.import_module('wandb')
except ImportError:
    wandb = None


def split_indices(indices, num_clients, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(indices), generator=g).tolist()
    shuffled = [indices[i] for i in perm]
    splits = []
    size = len(shuffled) // num_clients
    for i in range(num_clients):
        start = i * size
        end = (i + 1) * size if i < num_clients - 1 else len(shuffled)
        splits.append(shuffled[start:end])
    return splits


def fedavg_state_dicts(state_dicts, weights):
    total_weight = float(sum(weights))
    averaged = {}
    for key in state_dicts[0].keys():
        value = state_dicts[0][key]
        if torch.is_floating_point(value):
            stacked = torch.stack([sd[key].float() * w for sd, w in zip(state_dicts, weights)], dim=0)
            averaged[key] = (stacked.sum(dim=0) / total_weight).type_as(value)
        else:
            averaged[key] = value
    return averaged


def main(params):
    random_seed(params.seed)
    device = torch.device(f'cuda:{params.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    num_clients = 3

    images_dir = os.path.join(params.data_root, 'CelebA-HQ-img')
    labels_dir = os.path.join(params.data_root, 'CelebAMask-HQ-mask-anno')

    base_dataset = CelebAMaskHQ(images_dir, labels_dir, transform=DefaultTransform())
    total_len = len(base_dataset)
    train_indices, val_indices = load_or_create_split(
        params.split_file, total_len, val_ratio=params.val_ratio, seed=params.seed
    )

    train_dataset = CelebAMaskHQ(images_dir, labels_dir, transform=TrainTransform(image_size=params.image_size))
    val_dataset = Subset(
        CelebAMaskHQ(images_dir, labels_dir, transform=DefaultTransform()),
        val_indices,
    )

    client_splits = split_indices(train_indices, num_clients, seed=params.seed)
    client_loaders = []
    for client_idx, indices in enumerate(client_splits):
        client_dataset = Subset(train_dataset, indices)
        loader = DataLoader(
            client_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        client_loaders.append(loader)
        print(f'Client {client_idx}: {len(client_dataset)} samples')

    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f'Train dataset size: {len(train_indices)}')
    print(f'Val dataset size: {len(val_dataset)}')

    global_model = BiSeNet(num_classes=params.num_classes, backbone_name=params.backbone).to(device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = params.wandb_run_name or 'run'
    run_name = f'{run_name}_{timestamp}'

    if params.wandb:
        if wandb is None:
            raise ImportError('wandb is not installed. Run: pip install wandb')
        wandb.init(
            project=params.wandb_project,
            entity=params.wandb_entity,
            name=run_name,
            config=vars(params),
        )

    n_min = params.batch_size * params.image_size[0] * params.image_size[1] // 16
    criterion = OhemLossWrapper(thresh=params.score_thres, min_kept=n_min)

    best_miou = 0.0
    os.makedirs(f'./weights/{timestamp}', exist_ok=True)

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
    client_models = []
    client_optimizers = []
    client_lr_schedulers = []

    for client_idx in range(num_clients):
        client_model = BiSeNet(num_classes=params.num_classes, backbone_name=params.backbone).to(device)
        parameters = add_weight_decay(client_model, params.weight_decay)
        client_optimizers.append(torch.optim.SGD(
            parameters, lr=params.lr_start, momentum=params.momentum, weight_decay=params.weight_decay
        ))
        client_lr_schedulers.append(PolynomialLR(
            client_optimizers[client_idx], total_iters=len(client_loaders[client_idx]) * (params.epochs - params.lr_warmup_epochs), power=params.power
        ))
        client_models.append(client_model)


    global_step = 0
    for epoch in range(params.epochs):
        client_states = []
        client_sizes = []

        for client_idx, client_loader in enumerate(client_loaders):
            print(f"Client {client_idx} training ...")
            client_models[client_idx].load_state_dict(global_model.state_dict())
            client_models[client_idx].train()

            steps_in_client = len(client_loader)
            train_one_epoch(
                client_models[client_idx],
                criterion,
                client_optimizers[client_idx],
                client_loader,
                client_lr_schedulers[client_idx],
                device,
                epoch,
                params.print_freq,
                scaler=None,
                step_base=global_step,
            )
            global_step += steps_in_client

            client_states.append({k: v.detach().cpu() for k, v in client_models[client_idx].state_dict().items()})
            client_sizes.append(len(client_loader.dataset))

        averaged_state = fedavg_state_dicts(client_states, client_sizes)
        global_model.load_state_dict(averaged_state)

        step = global_step
        metrics = evaluate(
            global_model,
            criterion,
            val_loader,
            device,
            params.num_classes,
            class_names=class_names,
            print_freq=params.print_freq,
            log_images=params.wandb and params.wandb_log_images,
            image_count=params.wandb_image_count,
            log_step=step,
            epoch=epoch,
            vis_dir=os.path.join('visualization', run_name),
        )

        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    'val/loss': metrics['val/loss'],
                    'val/miou': metrics['val/miou'],
                    'val/macro_f1': metrics['val/macro_f1'],
                    'val/micro_f1': metrics['val/micro_f1'],
                },
                step=step,
            )
            per_iou = metrics['val/per_iou']
            per_f1 = metrics['val/per_f1']
            for idx, name in enumerate(metrics['val/class_names']):
                wandb.log(
                    {f'val/iou/{name}': per_iou[idx], f'val/f1/{name}': per_f1[idx]},
                    step=step,
                )

        ckpt = {
            'model': global_model.state_dict(),
            'epoch': epoch,
        }
        torch.save(ckpt, f'./weights/{timestamp}/{params.backbone}_global.ckpt')

        if metrics['val/miou'] > best_miou:
            best_miou = metrics['val/miou']
            torch.save(global_model.state_dict(), f'./weights/{timestamp}/best_{params.backbone}_global.pt')
            print(f'Best mIoU updated: {best_miou:.6f}')

    torch.save(global_model.state_dict(), f'./weights/{timestamp}/{params.backbone}_global.pt')
    print(f'Final mIoU: {metrics["val/miou"]:.6f}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
