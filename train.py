import os
from datetime import datetime
import torch
import importlib
from pathlib import Path
try:
    wandb = importlib.import_module('wandb')
except ImportError:
    wandb = None

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import PolynomialLR

from models.bisenet import BiSeNet
from utils.loss import OhemLossWrapper
from utils.transform import TrainTransform, DefaultTransform

from utils.dataset import CelebAMaskHQ, load_or_create_split
from utils.utils import parse_args, random_seed
from utils.function import train_one_epoch, evaluate, add_weight_decay
from utils.lr_scheduler import WarmupPolyLrScheduler
from utils.celebamask_hq import CelebAMaskHQ as CelebAMaskHQ_new


def main(params):
    random_seed(params.seed)
    device = torch.device(f'cuda:{params.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # images_dir = os.path.join(params.data_root, 'CelebA-HQ-img')
    # labels_dir = os.path.join(params.data_root, 'CelebAMask-HQ-mask-anno')

    # base_dataset = CelebAMaskHQ(images_dir, labels_dir, transform=DefaultTransform())
    # total_len = len(base_dataset)
    # train_indices, val_indices = load_or_create_split(
    #     params.split_file, total_len, val_ratio=params.val_ratio, seed=params.seed
    # )

    # train_dataset = Subset(
    #     CelebAMaskHQ(images_dir, labels_dir, transform=TrainTransform(image_size=params.image_size)),
    #     train_indices,
    # )
    # val_dataset = Subset(
    #     CelebAMaskHQ(images_dir, labels_dir, transform=DefaultTransform()),
    #     val_indices,
    # )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=params.batch_size,
    #     shuffle=True,
    #     num_workers=params.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=params.batch_size,
    #     shuffle=False,
    #     num_workers=params.num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    # )

    # ------------------------------------------------------------
    data_path = Path(params.data_root)

    train_dataset = CelebAMaskHQ_new(
        data_path,
        'train',
        resolution=params.image_size
    )
    val_dataset = CelebAMaskHQ_new(
        data_path,
        'val',
        resolution=params.image_size
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # ------------------------------------------------------------

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')

    # model
    model = BiSeNet(num_classes=params.num_classes, backbone_name=params.backbone)
    model.to(device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if params.wandb:
        if wandb is None:
            raise ImportError('wandb is not installed. Run: pip install wandb')
        run_name = params.wandb_run_name or 'run'
        run_name = f'{run_name}_{timestamp}'
        wandb.init(
            project=params.wandb_project,
            entity=params.wandb_entity,
            name=run_name,
            config=vars(params),
        )

    ignore_index = 0 if params.ignore_background else None
    n_min = params.batch_size * params.image_size[0] * params.image_size[1] // 16
    criterion = OhemLossWrapper(thresh=params.score_thres, min_kept=n_min, ignore_index=ignore_index)

    # optimizer
    parameters = add_weight_decay(model, params.weight_decay)
    optimizer = torch.optim.SGD(
        parameters, lr=params.lr_start, momentum=params.momentum, weight_decay=params.weight_decay
    )

    iters_per_epoch = len(train_loader)
    # lr_scheduler = PolynomialLR(
    #     optimizer, total_iters=iters_per_epoch * (params.epochs - params.lr_warmup_epochs), power=params.power
    # )
    lr_scheduler = WarmupPolyLrScheduler(
        optimizer,
        power=params.power, # 0.9
        max_iter=iters_per_epoch * params.epochs,
        warmup_iter=iters_per_epoch * params.lr_warmup_epochs,
        warmup_ratio=0.1,
        warmup='exp',
    )
    start_epoch = 0
    if params.resume:
        checkpoint = torch.load(f'./weights/{timestamp}/{params.backbone}.ckpt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    best_miou = 0.0
    (Path('./weights') / timestamp).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, params.epochs):
        step_base = epoch * len(train_loader)
        train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            lr_scheduler,
            device,
            epoch,
            params.print_freq,
            scaler=None,
            step_base=step_base,
        )
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
        step = (epoch + 1) * len(train_loader)
        metrics = evaluate(
            model,
            criterion,
            val_loader,
            device,
            params.num_classes,
            lb_ignore=0 if params.ignore_background else 255,
            class_names=class_names,
            print_freq=params.print_freq,
            log_images=params.wandb and params.wandb_log_images,
            image_count=params.wandb_image_count,
            log_step=step,
            epoch=epoch,
            vis_dir=os.path.join('visualization', f'{params.wandb_run_name or "run"}'),
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
            class_names = metrics['val/class_names']
            for idx, name in enumerate(class_names):
                wandb.log(
                    {
                        f'val/iou/{name}': per_iou[idx],
                        f'val/f1/{name}': per_f1[idx],
                    },
                    step=step,
                )

        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(ckpt, f'./weights/{timestamp}/{params.backbone}.ckpt')

        if metrics['val/miou'] > best_miou:
            best_miou = metrics['val/miou']
            torch.save(model.state_dict(), f'./weights/{timestamp}/best_{params.backbone}.pt')
            print(f'Best mIoU updated: {best_miou:.6f}')

    #  save final model
    state = model.state_dict()
    torch.save(state, f'./weights/{timestamp}/{params.backbone}.pt')
    print(f"Final mIoU: {metrics['val/miou']:.6f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
