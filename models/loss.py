import torch
import torch.nn as nn


class MSE_logit_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, targets):
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1.0 - 1e-4)
        pos_id = (targets == 1.0).float()
        neg_id = (targets == 0.0).float()
        pos_loss = pos_id * (pred - targets) ** 2
        neg_loss = neg_id * (pred) ** 2
        loss = 5.0 * pos_loss + 1.0 * neg_loss
        return loss


def compute_loss(conf_pred, class_pred, boxes_pred, targets):
    batch_size = conf_pred.size(0)
    conf_loss_function = MSE_logit_loss()
    class_loss_function = nn.CrossEntropyLoss(reduction="none")
    txty_loss_function = nn.BCEWithLogitsLoss(reduction="none")
    twth_loss_function = nn.MSELoss(reduction="none")

    conf_pred = conf_pred[:, :, 0]
    class_pred = class_pred.permute(0, 2, 1)
    txty_pred = boxes_pred[:, :, :2]
    twth_pred = boxes_pred[:, :, 2:]

    gt_obj = targets[:, :, 0]
    gt_class = targets[:, :, 1].long()
    gt_txty = targets[:, :, 2:4]
    gt_twth = targets[:, :, 4:6]
    gt_weight = targets[:, :, 6]

    conf_loss = conf_loss_function(conf_pred, gt_obj)
    conf_loss = conf_loss.sum() / batch_size

    class_loss = class_loss_function(class_pred, gt_class) * gt_obj
    class_loss = class_loss.sum() / batch_size

    txty_loss = txty_loss_function(txty_pred, gt_txty).sum(dim=-1) * gt_obj * gt_weight
    txty_loss = txty_loss.sum() / batch_size

    twth_loss = twth_loss_function(twth_pred, gt_twth).sum(dim=-1) * gt_obj * gt_weight
    twth_loss = twth_loss.sum() / batch_size

    bbox_loss = txty_loss + twth_loss

    total_loss = conf_loss + class_loss + bbox_loss

    return conf_loss, class_loss, bbox_loss, total_loss
