import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as npy

device = torch.device('cuda:3')
filename = "86.jpg"

path = "datasets/test"

images_test = list(filter(
    lambda x: x.endswith(".jpg") or x.endswith(".JPG") or x.endswith(".png"),
    os.listdir(path)
))

model = torch.load("models/checkpoint/detector-2.pth", map_location="cpu").to(device)
model.eval()

for filename in images_test:
    img_path = os.path.join(path, filename)
    img = Image.open(img_path)
    img = npy.array(img.resize((256, 256), Image.LANCZOS)).astype(npy.float32) * 2.0 / 255.0 - 1

    ts = transforms.ToTensor()(img).to(device)

    out = model([ts])

    img = npy.array(Image.open(img_path))
    origin_size = img.shape
    boxes = out[0]['boxes'].detach().cpu().numpy()
    scores = out[0]['scores'].detach().cpu().numpy()

    for i in range(boxes.shape[0]):
        if scores[i] < 0.05:
            break
        x1, y1, x2, y2 = boxes[i]
        x1 = int(x1 * origin_size[0] / 256)
        x2 = int(x2 * origin_size[0] / 256)
        y1 = int(y1 * origin_size[1] / 256)
        y2 = int(y2 * origin_size[1] / 256)
        img[y1 - 1 : y1 + 1, x1 - 1 : x2 + 1] = 255
        img[y2 - 1 : y2 + 1, x1 - 1 : x2 + 1] = 255
        img[y1 - 1 : y2 + 1, x1 - 1 : x1 + 1] = 255
        img[y1 - 1 : y2 + 1, x2 - 1 : x2 + 1] = 255

    print(filename, out)
    Image.fromarray(img).save("output/" + filename)