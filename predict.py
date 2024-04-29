import torch.nn as nn
import torch
import cv2
import numpy as np
from models.yolov1 import Yolo_V1
from torchvision.datasets import CocoDetection

path = "/home/dbp/dbp/project/yolov1/data/val2017/000000000139.jpg"
img = cv2.imread(path)
img = cv2.resize(img, (416, 416))
img = np.expand_dims(img, axis=0)
img = img.transpose((0, 3, 1, 2))
input_data = torch.tensor(img, dtype=torch.float32).to("cuda")

model = Yolo_V1("config/base.py", 416, "cuda", 2, 0.4, 0.7, False).to("cuda")
out = model(input_data)

print(out[2])
