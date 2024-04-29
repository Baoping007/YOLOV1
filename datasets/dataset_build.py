import os
import torch
import numpy as np
from .transform import BaseTransform, Augmentation
from .coco import COCODataset
from .cocoevaluate import COCOAPIEvaluator


def build_dataset(args, device, train_size, val_size):
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)  # BGR
    train_transform = Augmentation(train_size, pixel_mean, pixel_std)
    val_transform = BaseTransform(val_size, pixel_mean, pixel_std)
    if args.dataset == "coco":
        # data_root = os.path.join(args.root, "COCO")
        data_root = args.root
        num_classes = 80
        dataset = COCODataset(
            data_dir=data_root, img_size=train_size, transform=train_transform
        )

        evaluator = COCOAPIEvaluator(
            data_dir=data_root,
            img_size=val_size,
            device=device,
            transform=val_transform,
        )
    return dataset, num_classes, evaluator


def gt_create(input_size, stride, label_list=[]):
    batch_size = len(label_list)
    w, h = input_size, input_size
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, 1 + 1 + 4 + 1])
    for batch_index in range(batch_size):
        for gt_label in label_list[batch_index]:
            gt_class = int(gt_label[-1])
            res = generate_dxdywh(gt_label, w, h, s)
            if res:
                grid_x, grid_y, tx, ty, tw, th, weight = res
                if grid_x < ws and grid_y < hs:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = np.array(
                        [tx, ty, tw, th]
                    )
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight
    gt_tensor = gt_tensor.reshape(batch_size, -1, 1 + 1 + 4 + 1)
    return torch.from_numpy(gt_tensor).float()


def generate_dxdywh(gt_label, w, h, s):
    x1, y1, x2, y2 = gt_label[:-1]
    c_x = (x2 + x1) * w / 2
    c_y = (y2 + y1) * h / 2
    box_w = (x2 - x1) * w
    box_h = (y2 - y1) * h
    if box_w < 1e-4 or box_h < 1e-4:
        return False
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)

    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight
