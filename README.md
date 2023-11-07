# Mosaic-Recognizer

## Description

This repository provides a mosaic recognization model based on Fasterrcnn+Resnet50+FPN and using Torch.

## Usage

All package versions are just an example. You can use python 3.9, 3.8, 3.11, torch 1.12.0 or whatever you want.

First, create a new virtual environment.

```
conda create -n mosaic_recognizer python=3.10

conda activate mosaic_recognizer
```

Second, install numpy and PIL.

```
pip install numpy
pip install pillow
```

Next, install torch and torchvision.

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Then, download our model at 'Release' in this repository.

Finally, use it to recognize all mosaics in your picture.

For one single photo:
```
python recognizer.py --model path/to/model/filename.pth --threshold 0.1 --filename 1.png --output output
```

For a folder:
```
python recognizer.py --model path/to/model/filename.pth --threshold 0.1 --folder input --output output
```
