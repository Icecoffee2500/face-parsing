import torch
import torch.nn as nn


class OhemCELoss(nn.Module):
    def __init__(self, thresh: float, min_kept: int, ignore_index=None) -> None:
        super().__init__()
        self.thresh = torch.log(torch.tensor(1 / thresh, dtype=torch.float))
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        if ignore_index is None:
            self.criteria = nn.CrossEntropyLoss(reduction='none')
        else:
            self.criteria = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.criteria(logits, labels).view(-1)
        if self.ignore_index is not None:
            valid = labels.view(-1) != self.ignore_index
            loss = loss[valid]
        if loss.numel() == 0:
            return logits.sum() * 0.0

        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < self.min_kept:
            k = min(self.min_kept, loss.numel())
            loss_hard, _ = loss.topk(k)
        return torch.mean(loss_hard)


class OhemLossWrapper:
    def __init__(self, thresh: float, min_kept: int, ignore_index=None) -> None:
        self.loss = OhemCELoss(thresh=thresh, min_kept=min_kept, ignore_index=ignore_index)

    def __call__(self, output, labels):
        out, out16, out32 = output

        loss1 = self.loss(out, labels)
        loss2 = self.loss(out16, labels)
        loss3 = self.loss(out32, labels)

        loss = loss1 + loss2 + loss3
        return loss
