import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes,box_convert
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
import json
import pandas as pd
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Resize(size=(224,224),antialias=True))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        w,h=img.shape[1:]
        mask = read_image(mask_path)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)
        num_boxes=boxes.shape[0]
        target = torch.stack([boxes[:,0]/w,boxes[:,1]/h,boxes[:,2]/w,boxes[:,3]/h],axis=0)
        target=torch.transpose(target,0,1)
        img=img.repeat(num_boxes,1,1,1)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def PennFudanDataLoader(train,batch_size):
    dataset=PennFudanDataset('PennFudanPed',get_transform(train=train))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    return data_loader

class TACODataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        annotations = json.load(open(os.path.join(self.root, "data", "annotations.json"),"r"))
        df_images=pd.DataFrame(annotations['images'])
        self.imgs = df_images.file_name.to_list()
        self.annotations=pd.DataFrame(annotations['annotations'])

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "data", self.imgs[idx])
        img = read_image(img_path)
        w,h=img.shape[1:]
        boxes = self.annotations.loc[self.annotations.image_id==idx].bbox.values
        # get bounding box coordinates for each mask
        boxes = np.stack([b for b in boxes])
        boxes=box_convert(torch.Tensor(boxes), 'xywh', 'xyxy')
        num_boxes=boxes.shape[0]
        target = torch.stack([boxes[:,0]/w,boxes[:,1]/h,boxes[:,2]/w,boxes[:,3]/h],axis=0)
        target=torch.transpose(target,0,1)
        img=img.repeat(num_boxes,1,1,1)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def TACODataLoader(train,batch_size):
    dataset=TACODataset('/media/sergio/45B9-67F2/code/TACO',get_transform(train=train))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    return data_loader