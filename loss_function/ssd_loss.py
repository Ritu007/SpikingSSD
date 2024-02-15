import torch
import torch.nn as nn
import torch.nn.functional as F


class SSDLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3, n_neg_min=0, alpha=1.0):
        super(SSDLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        absolute_loss = torch.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = torch.where(torch.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return torch.sum(l1_loss, dim=-1)

    def log_loss(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-15)  # Avoid zeros
        log_loss = -torch.sum(y_true * torch.log(y_pred), dim=-1)
        return log_loss

    def forward(self, y_true, y_pred):
        batch_size = y_pred.size(0)
        n_boxes = y_pred.size(1)

        # Compute classification and localization losses
        classification_loss = self.log_loss(y_true[:, :, :-12], y_pred[:, :, :-12])
        localization_loss = self.smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8])

        # Compute positive and negative losses
        negatives = y_true[:, :, 0]
        positives = torch.max(y_true[:, :, 1:-12], dim=-1)[0]
        n_positive = torch.sum(positives)

        pos_class_loss = torch.sum(classification_loss * positives, dim=-1)
        neg_class_loss_all = classification_loss * negatives
        n_neg_losses = torch.count_nonzero(neg_class_loss_all, dim=-1)

        def f1():
            return torch.zeros(batch_size)

        def f2():
            neg_class_loss_all_1D = neg_class_loss_all.view(batch_size * n_boxes)
            _, indices = torch.topk(neg_class_loss_all_1D, k=n_neg_losses.item())
            negatives_keep = torch.zeros(batch_size * n_boxes)
            negatives_keep[indices] = 1
            negatives_keep = negatives_keep.view(batch_size, n_boxes)
            neg_class_loss = torch.sum(classification_loss * negatives_keep, dim=-1)
            return neg_class_loss

        neg_class_loss = torch.where(n_neg_losses == 0, f1(), f2())

        class_loss = pos_class_loss + neg_class_loss
        loc_loss = torch.sum(localization_loss * positives, dim=-1)

        total_loss = (class_loss + self.alpha * loc_loss) / max(1.0, n_positive.item())
        total_loss = total_loss * batch_size

        return total_loss
