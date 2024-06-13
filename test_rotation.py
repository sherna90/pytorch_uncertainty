import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes,box_convert
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
import json
import pandas as pd
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt

root="PennFudanPed"
idx=100
imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
img_path = os.path.join(root, "PNGImages", imgs[idx])
mask_path = os.path.join(root, "PedMasks", masks[idx])
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
mask = cv2.imread(mask_path)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rotated_boxes=list()
for cnt in contours:
    if cnt.shape[0]>0:
        (x, y), (w, h), angle = cv2.minAreaRect(cnt)
        rotated_boxes.append(((x, y), (w, h), angle))
for rect in rotated_boxes:
    box = cv2.boxPoints(rect) 
    box = np.int_(box) 
    img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
plt.imshow(img)
plt.show()
 

