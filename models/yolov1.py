import numpy as np
import torch
import torch.nn as nn
from .backbone import build_resnet
from .neck import neck
from .head import head
from .loss import compute_loss


class Yolo_V1(nn.Module):
    def __init__(
        self,
        device,
        img_size,
        num_classes,
        conf_th,
        nms_th,
        train,
    ):
        super().__init__()
        self.img_size = img_size
        self.device = device
        self.num_classes = num_classes
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.is_train = train
        self.stride = 32
        self.backbone, feat_dim = build_resnet("resnet18", pretrained=train)
        self.neck = neck(feat_dim)
        self.head = head(feat_dim, num_classes)
        self.create_grid((img_size, img_size))

        if self.is_train:
            self.head.init_bias()

    def create_grid(self, size):
        w, h = size
        nw, nh = w // self.stride, h // self.stride
        X, Y = torch.meshgrid(torch.arange(nh), torch.arange(nw))
        grid = torch.stack((X, Y), dim=-1).float()
        grid = grid.view(-1, 2).to(self.device)
        self.grid = grid

    def decode_xywh(self, pred):
        """将模型输出cxcytwth转化为x1y1x2y2"""
        output = torch.zeros_like(pred)

        pred[..., :2] = torch.sigmoid(pred[..., :2]) + self.grid
        pred[..., 2:] = torch.exp(pred[..., 2:])
        output[..., :2] = pred[..., :2] * self.stride - pred[..., 2:]
        output[..., 2:] = pred[..., :2] * self.stride + pred[..., 2:]
        return output

    def post_process(self, scores, boxes):
        """
        Input:
          boxes: [H*W, 4]
          scores: [H*W, num_classes]
        Output:
          boxes:
        """
        labels = np.argmax(scores, axis=-1)
        scores = scores[(np.arange(scores.shape[0]), labels)]

        index = scores > self.conf_th
        labels = labels[index]
        boxes = boxes[index]
        scores = scores[index]

        keep = np.zeros(boxes.shape[0], dtype=np.int32)
        for i in range(self.num_classes):
            ind = np.where(labels == i)[0]
            if len(ind) == 0:
                continue
            t_boxes = boxes[ind]
            t_scores = scores[ind]
            k = self.nms(t_boxes, t_scores)
            keep[ind[k]] = 1
        keep = np.where(keep == 1)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        return labels, scores, boxes

    def nms(self, boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (y2 - y1) * (x2 - x1)
        order = np.argsort(scores)[::-1]
        keep = []
        while len(order):
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(xx2 - xx1, 0)
            h = np.maximum(yy2 - yy1, 0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= self.nms_th]

        return keep

    @torch.no_grad()
    def inference(self, x):
        feat = self.backbone(x)

        feat = self.neck(feat)

        pred = self.head(feat)
        pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        conf_pred = pred[..., :1]
        class_pred = pred[..., 1 : 1 + self.num_classes]
        boxes_pred = pred[..., 1 + self.num_classes :]

        """
        默认batch为1
        """
        conf_pred = conf_pred[0]
        class_pred = class_pred[0]
        boxes_pred = boxes_pred[0]

        scores = torch.sigmoid(conf_pred) * torch.softmax(class_pred, dim=-1)
        boxes = self.decode_xywh(boxes_pred) / self.img_size
        boxes = torch.clamp(boxes, min=0, max=1)

        scores = scores.to("cpu").numpy()
        boxes = boxes.to("cpu").numpy()
        b_labels, b_scores, b_boxes = self.post_process(scores, boxes)
        return b_boxes, b_scores, b_labels

    def forward(self, x, targets=None):
        if not self.is_train:
            return self.inference(x)
        else:
            feat = self.backbone(x)
            feat = self.neck(feat)
            pred = self.head(feat)
            pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            conf_pred = pred[..., :1]
            class_pred = pred[..., 1 : 1 + self.num_classes]
            boxes_pred = pred[..., 1 + self.num_classes :]
            conf_loss, cls_loss, boxes_loss, total_loss = compute_loss(
                conf_pred, class_pred, boxes_pred, targets
            )
            return conf_loss, cls_loss, boxes_loss, total_loss
