import torch
import torch.nn as nn
import numpy as np
import time
import wandb

def add_weight_decay(model, weight_decay=1e-5):
    """Applying weight decay to only weights, not biases"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias') or isinstance(param, nn.BatchNorm2d) or 'bn' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': weight_decay}]


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    lr_scheduler,
    device,
    epoch,
    print_freq,
    scaler=None,
    step_base=0,
):
    model.train()
    batch_loss = []
    for batch_idx, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image = image.to(device)
        target = target.to(device)

        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            output = model(image)
            # print(f"out shape: {output[0].shape}")
            # print(f"out16 shape: {output[1].shape}")
            # print(f"out32 shape: {output[2].shape}")

            # print(f"output unique values: {torch.unique(output[0])}")
            # print(f"output16 unique values: {torch.unique(output[1])}")
            # print(f"output32 unique values: {torch.unique(output[2])}")

            # print(f"target shape: {target.shape}")
            # print(f"target unique values: {torch.unique(target)}")
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        batch_loss.append(loss.item())

        if (batch_idx + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f'Train: [{epoch:>3d}][{batch_idx + 1:>4d}/{len(data_loader)}] '
                f'Loss: {loss.item():.4f}  '
                f'Time: {(time.time() - start_time):.3f}s '
                f'LR: {lr:.7f} '
            )
            if wandb is not None and wandb.run is not None:
                step = step_base + batch_idx + 1
                wandb.log({'train/loss': loss.item(), 'train/lr': lr}, step=step)
    print(f'Avg batch loss: {np.mean(batch_loss):.7f}')


def evaluate(
    model,
    criterion,
    data_loader,
    device,
    n_classes,
    lb_ignore=255,
    class_names=None,
    print_freq=50,
    log_images=False,
    image_count=4,
    log_step=None,
    epoch=None,
    vis_dir=None,
):
    model.eval()
    losses = []
    confusion = torch.zeros((n_classes, n_classes), device=device)
    vis_grid = None
    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(data_loader):
            start_time = time.time()
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output, target)
            losses.append(loss.item())
            if (batch_idx + 1) % print_freq == 0:
                print(
                    f'Val: [{batch_idx + 1:>4d}/{len(data_loader)}] '
                    f'Loss: {loss.item():.4f}  '
                    f'Time: {(time.time() - start_time):.3f}s'
                )
            logits = output[0] if isinstance(output, (list, tuple)) else output
            preds_full = torch.argmax(logits, dim=1)
            keep = target != lb_ignore
            preds = preds_full[keep]
            label = target[keep]
            confusion += torch.bincount(
                label * n_classes + preds,
                minlength=n_classes ** 2,
            ).reshape(n_classes, n_classes)
            if batch_idx == 0 and vis_dir and epoch is not None:
                palette = [
                    (0, 0, 0),
                    (255, 204, 204),
                    (255, 102, 102),
                    (255, 153, 51),
                    (255, 255, 0),
                    (102, 255, 102),
                    (102, 204, 255),
                    (102, 102, 255),
                    (204, 102, 255),
                    (255, 102, 204),
                    (153, 102, 51),
                    (255, 153, 153),
                    (255, 204, 153),
                    (204, 255, 153),
                    (153, 255, 204),
                    (153, 204, 255),
                    (204, 153, 255),
                    (255, 153, 255),
                    (192, 192, 192),
                ]
                mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
                imgs = (image * std + mean).clamp(0, 1).detach().cpu().numpy()
                preds_np = preds_full.detach().cpu().numpy()
                targets_np = target.detach().cpu().numpy()
                count = min(image_count, imgs.shape[0])
                rows = []
                for i in range(count):
                    img_np = (imgs[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt_np = targets_np[i].astype(np.int32)
                    pr_np = preds_np[i].astype(np.int32)

                    gt_color = np.zeros_like(img_np)
                    pr_color = np.zeros_like(img_np)
                    for cls_id in range(1, min(n_classes, len(palette))):
                        gt_color[gt_np == cls_id] = palette[cls_id]
                        pr_color[pr_np == cls_id] = palette[cls_id]

                    alpha = 0.5
                    gt_overlay = (img_np * (1 - alpha) + gt_color * alpha).astype(np.uint8)
                    pr_overlay = (img_np * (1 - alpha) + pr_color * alpha).astype(np.uint8)
                    row = np.concatenate([img_np, gt_overlay, pr_overlay], axis=1)
                    rows.append(row)

                grid = np.concatenate(rows, axis=0)
                vis_grid = grid
                # os.makedirs(vis_dir, exist_ok=True)
                # out_path = os.path.join(vis_dir, f'{epoch}_vis.jpg')
                # Image.fromarray(grid).save(out_path)

            if (
                log_images
                and wandb is not None
                and wandb.run is not None
                and batch_idx == 0
                and vis_grid is not None
            ):
                img_log = wandb.Image(vis_grid, caption=f'epoch {epoch}')
                if log_step is None:
                    wandb.log({'val/samples': img_log})
                else:
                    wandb.log({'val/samples': img_log}, step=log_step)
    tps = confusion.diag()
    fps = confusion.sum(dim=0) - tps
    fns = confusion.sum(dim=1) - tps

    ious = tps / (tps + fps + fns + 1)
    miou = ious.nanmean().item()

    macro_precision = tps / (tps + fps + 1)
    macro_recall = tps / (tps + fns + 1)
    f1_scores = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall + 1e-6)
    macro_f1 = f1_scores.nanmean().item()

    tps_ = tps.sum()
    fps_ = fps.sum()
    fns_ = fns.sum()
    micro_precision = tps_ / (tps_ + fps_ + 1)
    micro_recall = tps_ / (tps_ + fns_ + 1)
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-6)).item()

    print(
        f'Val loss: {np.mean(losses):.7f} | '
        f'mIoU: {miou:.6f} | '
        f'macro F1: {macro_f1:.6f} | '
        f'micro F1: {micro_f1:.6f}'
    )
    per_iou = ious.detach().cpu().numpy()
    per_f1 = f1_scores.detach().cpu().numpy()
    names = class_names or [f'class_{idx:02d}' for idx in range(n_classes)]
    if len(names) != n_classes:
        names = [f'class_{idx:02d}' for idx in range(n_classes)]
    for idx, name in enumerate(names):
        iou_val = per_iou[idx]
        f1_val = per_f1[idx]
        print(f'{name} | IoU: {iou_val:.6f} | F1: {f1_val:.6f}')

    return {
        'val/loss': float(np.mean(losses)),
        'val/miou': miou,
        'val/macro_f1': macro_f1,
        'val/micro_f1': micro_f1,
        'val/per_iou': per_iou,
        'val/per_f1': per_f1,
        'val/class_names': names,
    }