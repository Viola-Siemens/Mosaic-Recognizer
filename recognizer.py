import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as npy
from typing import List


def single_output(origin: npy.ndarray, boxes: npy.ndarray, scores: npy.ndarray, output_filename: str, verbose: bool, threshold: float):
    origin_size = origin.shape

    for b in range(boxes.shape[0]):
        if scores[b] < threshold:
            continue
        x1, y1, x2, y2 = boxes[b]
        x1 = int(x1 * origin_size[0] / 256)
        x2 = int(x2 * origin_size[0] / 256)
        y1 = int(y1 * origin_size[1] / 256)
        y2 = int(y2 * origin_size[1] / 256)
        origin[y1 - 1 : y1 + 1, x1 - 1 : x2 + 1] = 255
        origin[y2 - 1 : y2 + 1, x1 - 1 : x2 + 1] = 255
        origin[y1 - 1 : y2 + 1, x1 - 1 : x1 + 1] = 255
        origin[y1 - 1 : y2 + 1, x2 - 1 : x2 + 1] = 255

    Image.fromarray(origin).save(output_filename)


def test_folder(model: torch.nn.Module, device: torch.device, path: str, output_path: str, verbose: bool, threshold: float, batch_size: int):
    images_test = list(filter(
        lambda x: x.endswith(".jpg") or x.endswith(".jpeg") or x.endswith(".png") or x.endswith(".JPG") or x.endswith(".PNG"),
        os.listdir(path)
    ))

    filenames = []
    imgs = []
    inputs = []
    for filename in images_test:
        img_path = os.path.join(path, filename)
        img = Image.open(img_path)

        filenames.append(filename)
        imgs.append(npy.array(img))
        img = npy.array(img.resize((256, 256), Image.LANCZOS)).astype(npy.float32) * 2.0 / 255.0 - 1
        ts = transforms.ToTensor()(img).to(device)
        if ts.shape[0] > 3:
            raise ValueError("Invalid image:", filename)
        inputs.append(ts)

        if len(inputs) >= batch_size:
            out = model(inputs)
            for i in range(len(inputs)):
                img = imgs[i]
                boxes = out[i]['boxes'].detach().cpu().numpy()
                scores = out[i]['scores'].detach().cpu().numpy()
                
                if verbose:
                    print(filenames[i], out[i])

                single_output(img, boxes, scores, os.path.join(output_path, filenames[i]), verbose, threshold)
            inputs = []
    if len(inputs) >= 0:
        out = model(inputs)
        for i in range(len(inputs)):
            img = imgs[i]
            boxes = out[i]['boxes'].detach().cpu().numpy()
            scores = out[i]['scores'].detach().cpu().numpy()
            
            if verbose:
                print(filenames[i], out[i])
                
            single_output(img, boxes, scores, os.path.join(output_path, filenames[i]), verbose, threshold)

def test_single(model: torch.nn.Module, device: torch.device, img_path: str, output_path: str, verbose: bool, threshold: float):
    img = Image.open(img_path)
    ts = transforms.ToTensor()(npy.array(img.resize((256, 256), Image.LANCZOS)).astype(npy.float32) * 2.0 / 255.0 - 1).to(device)

    out = model([ts])
    boxes = out[0]['boxes'].detach().cpu().numpy()
    scores = out[0]['scores'].detach().cpu().numpy()

    if verbose:
        print(filenames[0], out[0])

    single_output(npy.array(img), boxes, scores, os.path.join(output_path, os.path.basename(img_path)), verbose, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/detector.pth",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Threshold for showing bounding box. Bigger value may reduce duplicated boxes, while smaller value may find uncommon mosaics.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename of the image you want to recognize. Use '--folder' if you need to work with multiple images.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder for the images you want to recognize. Use '--filename' if you need to work with a single image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="The output folder.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for testing, only activated when using '--folder'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device for the model. Eg. cuda, cuda:0, cuda:1, cpu.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Set True to print model output to your terminal.",
    )

    opt = parser.parse_args()

    if opt.filename is None and opt.folder is None:
        raise ValueError("Both filename and folder are None!")

    device = torch.device(opt.device)
    model = torch.load(opt.model, map_location="cpu").to(device)
    model.eval()

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    if opt.folder is not None:
        test_folder(model, device, opt.folder, opt.output, opt.verbose, opt.threshold, opt.batch_size)
        print("Output folder:", opt.output)
    if opt.filename is not None:
        test_single(model, device, opt.filename, opt.output, opt.verbose, opt.threshold)
        print("Output filename:", os.path.join(opt.output, os.path.basename(opt.filename)))

    print("Enjoy!")
