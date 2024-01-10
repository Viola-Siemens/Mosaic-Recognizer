import os
import io
import re
import random
import imageio
from typing import List, Tuple, Callable
import scipy.stats as stats

import numpy as npy
from PIL import Image
import torch
import torchvision
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_ as clip_grad
from torch.optim import Optimizer, Adam, RMSprop
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from torchvision.ops import box_iou
from torchvision.utils import save_image
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

guassian_kernels = {}

for sigma_2 in range(4, 10):
    t = {}
    for radius in range(0, 32):
        total = 0
        sz = 2 * radius + 1
        kernel = npy.zeros((sz, sz, 3))
        for i in range(sz):
            for j in range(sz):
                kernel[i, j, :] = npy.exp(-((i - radius) ** 2 + (j - radius) ** 2) / (2 * sigma_2)) / (2 * npy.pi * sigma_2)
        t[radius] = kernel / kernel.sum() * 3
    guassian_kernels[sigma_2] = t

def mosaic1(img: npy.ndarray, radius: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    for x in range(x1, x2):
        for y in range(y1, y2):
            img[y, x] = img[y // radius * radius, x // radius * radius]

def mosaic2(img: npy.ndarray, radius: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    for x in range(0, x2 - x1, radius):
        for y in range(0, y2 - y1, radius):
            img[y1 + y : min(y1 + y + radius, y2), x1 + x : min(x1 + x + radius, x2)] = img[y1 + y : min(y1 + y + radius, y2), x1 + x : min(x1 + x + radius, x2)].mean(axis=(0, 1))

def color1(img: npy.ndarray, radius: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int, color: Tuple[int, int, int]):
    img[y1 : y2, x1 : x2] = color

def blur1(img: npy.ndarray, radius: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    sigma_2 = random.randint(4, 9)
    for x in range(x1, x2):
        for y in range(y1, y2):
            r = min(radius, x, w - 1 - x, y, h - 1 - y)
            img[y, x] = (guassian_kernels[sigma_2][r] * img[y - r : y + r + 1, x - r : x + r + 1]).sum(axis=(0, 1))

def blur2(img: npy.ndarray, radius: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    for x in range(x1, x2):
        for y in range(y1, y2):
            img[y, x] = img[max(0, y - radius) : min(y + radius + 1, h), max(0, x - radius) : min(x + radius + 1, w)].mean(axis=(0, 1))

def jpegBlur(im):
    buf = io.BytesIO()
    imageio.imwrite(buf, im, format='jpg',quality=random.randint(5, 75))
    s = buf.getbuffer()
    return imageio.imread(s, format='jpg')

def choose_and_mask(a: int, img: npy.ndarray, radius: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int, color: Tuple[int, int, int]):
    if a < 24:
        mosaic1(img, radius, x1, y1, x2, y2, w, h)
    elif a < 42:
        mosaic2(img, radius, x1, y1, x2, y2, w, h)
    elif a < 61:
        color1(img, radius, x1, y1, x2, y2, w, h, color)
    elif a < 81:
        blur1(img, radius, x1, y1, x2, y2, w, h)
    else:
        blur2(img, radius, x1, y1, x2, y2, w, h)
    if random.randint(0, 9) < 4:
        img = img + npy.random.laplace(0.0, random.randint(0, 24) / 100.0, img.shape)

def get_label(a: int):
    # 1: pixelize; 2: color; 3: blur
    if a < 42:
        return 1
    if a < 61:
        return 2
    return 3

def handle(img) -> Tuple[npy.ndarray, npy.ndarray, npy.ndarray]:
    w, h = img.size

    if w < 15 or h < 15:
        img = img.resize((w * 15, h * 15), Image.LANCZOS)
        w, h = img.size

    boxes = []
    bound = random.randint(0, 2) + random.randint(0, 2) + 1
    choose = random.randint(0, 99)
    color_stats = stats.beta.rvs(0.75, 0.75, size=3)
    color = (color_stats[0] * 2.0 - 1, color_stats[1] * 2.0 - 1, color_stats[2] * 2.0 - 1)
    radius = random.randint(0, 8) + random.randint(0, 7) + 5
    mask = torch.zeros((256, 256, 3), dtype=torch.uint8)

    img = npy.array(img).astype(npy.float32) * 2.0 / 255.0 - 1
    for _ in range(bound):
        mask_size = random.randint(0, 15) + random.randint(0, 25) + random.randint(0, 35) + 20
        xy_diff = random.randint(-10, 10) + random.randint(-5, 5)
        if mask_size >= w or mask_size + xy_diff >= h:
            mask_size = 12
            xy_diff = random.randint(-2, 2)

        std = 0
        tries = 100
        while std < 0.125:
            tries -= 1
            x1 = random.randint(0, w - mask_size)
            y1 = random.randint(0, h - mask_size - xy_diff)
            x2 = x1 + mask_size
            y2 = y1 + mask_size + xy_diff
            std = img[y1 : y2, x1 : x2].std(axis=(0, 1)).mean()
            box = [x1 * 256 / w, y1 * 256 / h, x2 * 256 / w, y2 * 256 / h]
            if len(boxes) > 0 and box_iou(torch.tensor(boxes), torch.tensor([box])).sum() >= 0.01:
                std = 0
            if tries <= 0:
                if len(boxes) > 0:
                    break
                else:
                    std = 1
        else:
            boxes.append(box)
            mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = 1
            choose_and_mask(choose, img, radius, x1, y1, x2, y2, w, h, color)
    boxes = npy.array(boxes)
    img_pil = npy.array(Image.fromarray(((img + 1) * 255.0 / 2.0).astype(npy.uint8)).resize((256, 256), Image.LANCZOS))
    if random.randint(0, 99) < 67:
        img_pil = jpegBlur(img_pil)
    img = img_pil.astype(npy.float32) * 2.0 / 255.0 - 1
    
    return img, mask, boxes, get_label(choose)

from pascal_voc_toolkit import XmlHandler
from pathlib import Path

def handle_anno(img, path_anno) -> Tuple[npy.ndarray, npy.ndarray, npy.ndarray]:
    if os.path.exists(path_anno) and random.randint(0, 2) == 0:
        w, h = img.size

        choose = random.randint(0, 99)
        color = (random.random() * 2.0 - 1, random.random() * 2.0 - 1, random.random() * 2.0 - 1)
        radius = random.randint(0, 8) + random.randint(0, 7) + 5
        mask = torch.zeros((256, 256, 3), dtype=torch.uint8)

        img = npy.array(img).astype(npy.float32) * 2.0 / 255.0 - 1

        xml = XmlHandler(Path(path_anno))
        boxes = []
        for l in xml.get_all_bbox_coordinates_list():
            x1, y1, x2, y2, _name = l
            box = [x1 * 256 / w, y1 * 256 / h, x2 * 256 / w, y2 * 256 / h]
            boxes.append(box)
            mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = 1
            choose_and_mask(choose, img, radius, x1, y1, x2, y2, w, h, color)
        boxes = npy.array(boxes)
        img_pil = npy.array(Image.fromarray(((img + 1) * 255.0 / 2.0).astype(npy.uint8)).resize((256, 256), Image.LANCZOS))
        if random.randint(0, 99) < 67:
            img_pil = jpegBlur(img_pil)
        img = img_pil.astype(npy.float32) * 2.0 / 255.0 - 1

        return img, mask, boxes, get_label(choose)
    return handle(img)

class CityScapesDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.images = self._get_images(os.path.join(path, "dataset"))
        self.transform = transform

    def __getitem__(self, item) -> Tuple:
        path_img = self.images[item]
        rn = random.randint(0, 1024)
        img = Image.open(path_img).convert("RGB").crop((rn, 0, rn + 1024, 1024))
        
        img, mask, boxes, label = handle(img)
        if self.transform is not None:
            img = self.transform(img)
        target = {
            "boxes": torch.tensor(boxes),
            "masks": mask,
            "image_id": path_img,
            "area": torch.tensor([] if boxes.shape[0] == 0 else (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])),
            "labels": label * torch.ones((boxes.shape[0],), dtype=torch.int64),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64)
        }
        return img, target

    def __len__(self):
        return len(self.images)

    def _get_images(self, path: str) -> List:
        images = []

        pattern = r'^[a-z]+$'

        labels = sorted(list(filter(
            lambda x: re.search(pattern, x) != None,
            os.listdir(path)
        )))

        for l in labels:
            p2 = os.path.join(path, l)
            images_train = list(filter(
                lambda x: x.endswith(".png"),
                os.listdir(p2)
            ))
            for im in images_train:
                images.append(os.path.join(p2, im))

        return images

class WoodScapeDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.images = self._get_images(os.path.join(path, "WoodScape/rgb_images"))
        self.transform = transform

    def __getitem__(self, item) -> Tuple:
        path_img = self.images[item]
        img = Image.open(path_img).convert("RGB")

        img, mask, boxes, label = handle(img)
        if self.transform is not None:
            img = self.transform(img)
        target = {
            "boxes": torch.tensor(boxes),
            "masks": mask,
            "image_id": path_img,
            "area": torch.tensor([] if boxes.shape[0] == 0 else (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])),
            "labels": label * torch.ones((boxes.shape[0],), dtype=torch.int64),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64)
        }
        return img, target

    def __len__(self):
        return len(self.images)

    def _get_images(self, path: str) -> List:
        images = []

        images_train = list(filter(
            lambda x: x.endswith(".png"),
            os.listdir(path)
        ))
        for im in images_train:
            images.append(os.path.join(path, im))

        return images

class Fisheye8KDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.images = self._get_images(os.path.join(path, "Fisheye8K"))
        self.transform = transform

    def __getitem__(self, item) -> Tuple:
        path_img = self.images[item]
        img = Image.open(path_img).convert("RGB")

        img, mask, boxes, label = handle(img)
        if self.transform is not None:
            img = self.transform(img)
        target = {
            "boxes": torch.tensor(boxes),
            "masks": mask,
            "image_id": path_img,
            "area": torch.tensor([] if boxes.shape[0] == 0 else (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])),
            "labels": label * torch.ones((boxes.shape[0],), dtype=torch.int64),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64)
        }
        return img, target

    def __len__(self):
        return len(self.images)

    def _get_images(self, path: str) -> List:
        images = []

        for x in ["train", "test"]:
            p2 = os.path.join(path, x + "/images")
            images_train = list(filter(
                lambda x: x.endswith(".png"),
                os.listdir(p2)
            ))
            for im in images_train:
                images.append(os.path.join(p2, im))

        return images

class IHAZEDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.images = self._get_images(os.path.join(path, "I-HAZE/# I-HAZY NTIRE 2018/hazy"))
        self.transform = transform

    def __getitem__(self, item) -> Tuple:
        path_img = self.images[item]
        img = Image.open(path_img).convert("RGB")
        w, h = img.size
        if w > 1200 and h > 1200:
            w = w // 4 * 4
            h = h // 4 * 4
            img = img.crop((0, 0, w, h)).resize((w // 4, h // 4), Image.LANCZOS)

        img, mask, boxes, label = handle(img)
        if self.transform is not None:
            img = self.transform(img)
        target = {
            "boxes": torch.tensor(boxes),
            "masks": mask,
            "image_id": path_img,
            "area": torch.tensor([] if boxes.shape[0] == 0 else (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])),
            "labels": label * torch.ones((boxes.shape[0],), dtype=torch.int64),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64)
        }
        return img, target

    def __len__(self):
        return len(self.images)

    def _get_images(self, path: str) -> List:
        images = []

        images_train = list(filter(
            lambda x: x.endswith(".jpg"),
            os.listdir(path)
        ))
        for im in images_train:
            images.append(os.path.join(path, im))

        return images

class ILSVRCDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.path = os.path.join(path, "ILSVRC")
        self.images = self._get_images()
        self.transform = transform

    def __getitem__(self, item) -> Tuple:
        path_img = self.images[item]
        path_anno = path_img.replace("Data", "Annotations").replace(".JPEG", ".xml")
        img = Image.open(path_img).convert("RGB")

        img, mask, boxes, label = handle_anno(img, path_anno)
        if self.transform is not None:
            img = self.transform(img)
        target = {
            "boxes": torch.tensor(boxes),
            "masks": mask,
            "image_id": path_img,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "labels": label * torch.ones((boxes.shape[0],), dtype=torch.int64),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64)
        }
        return img, target

    def __len__(self):
        return len(self.images)

    def _get_images(self) -> Tuple[List, List]:
        images = []

        pattern = r'^n[0-9]+$'

        for x in ["train"]:
            path1 = os.path.join(self.path, "Data/CLS-LOC/" + x)

            labels = sorted(list(filter(
                lambda x: re.search(pattern, x) != None,
                os.listdir(path1)
            )))

            for l in labels:
                p2 = os.path.join(path1, l)
                images_train = list(filter(
                    lambda x: x.endswith(".JPEG"),
                    os.listdir(p2)
                ))
                for im in images_train:
                    images.append(os.path.join(p2, im))

        return images

def test(data_iter):
    for imgs, targets in data_iter:
        for c in range(len(imgs)):
            img = (imgs[c].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 255.0 / 2.0
            boxes = targets[c]['boxes'].detach().cpu().numpy()

            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[i]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                img[y1 - 1 : y1 + 1, x1 - 1 : x2 + 1] = 255
                img[y2 - 1 : y2 + 1, x1 - 1 : x2 + 1] = 255
                img[y1 - 1 : y2 + 1, x1 - 1 : x1 + 1] = 255
                img[y1 - 1 : y2 + 1, x2 - 1 : x2 + 1] = 255
            
            Image.fromarray(img.astype(npy.uint8)).save("output/test/%d.png" % c)
        exit(0)

from detection.utils import collate_fn

dataset = ConcatDataset([
    CityScapesDataset("../pipeline", transform=transforms.ToTensor()),
    WoodScapeDataset("./datasets", transform=transforms.ToTensor()),
    Fisheye8KDataset("./datasets", transform=transforms.ToTensor()),
    IHAZEDataset("./datasets", transform=transforms.ToTensor()),
    ILSVRCDataset("./datasets", transform=transforms.ToTensor())
])
len_dataset = len(dataset)
len_train_dataset = int(len_dataset * 0.8) + 76
len_val_dataset = len_dataset - len_train_dataset
train_dataset, val_dataset = random_split(dataset=dataset, lengths=[len_train_dataset, len_val_dataset])
train_data_iter = DataLoader(train_dataset, batch_size=25, num_workers=8, shuffle=True, collate_fn=collate_fn)
val_data_iter = DataLoader(val_dataset, batch_size=25, num_workers=8, shuffle=False, collate_fn=collate_fn)
print("Loaded %d train images and %d val images."%(len(train_dataset), len(val_dataset)))

# test(train_data_iter)

device = torch.device('cuda')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2)
# model = torch.load("models/checkpoint/detector-2.pth")
# print("Let's use", torch.cuda.device_count(), "GPUs!")
# model = torch.nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-3, weight_decay=8e-5)

from detection.engine import train_one_epoch, evaluate

for epoch in range(5):
    train_one_epoch(model, optimizer, train_data_iter, device, epoch, print_freq=40)
    torch.save(model, "models/checkpoint/detector-%d.pth"%(epoch+1))
    # evaluate(model, val_data_iter, device=device, print_freq=10)

torch.save(model, "models/detector.pth")
