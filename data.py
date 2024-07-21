import os

from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class MyDataset(Dataset):
    def __init__(self, data_list: list = []) -> None:
        super().__init__()
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]
    
    def append(self, data, label):
        self.data.append((data, label))

def make_image(num: int, font: str, font_size: int) -> Image.Image:
    kernel = np.ones((2,2),np.uint8)
    image = Image.new("RGB", (80, 80), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font=font, size=font_size)
    text_width, text_height = draw.textsize(str(num), font=font)
    x = (80 - text_width) / 2
    y = (80 - text_height) / 2
    draw.text((x, y), str(num), fill=(255, 255, 255), font=font)
    img = image.convert("L")
    img = np.array(img)
    img = img.reshape((1, 80, 80))
    return img

numbers = list(range(10))
sizes = list(range(30, 121, 5))
font_path = './font'
font_files = os.listdir(font_path)
font_paths = [os.path.join(font_path, font) for font in font_files]

def train_data():
    dataset = MyDataset()
    for num in numbers:
        for font_path in font_paths:
            for size in sizes:
                image = make_image(num, font=font_path, font_size=size)
                dataset.append(data=torch.Tensor(image), label=torch.tensor(num))
    return dataset

test_sizes = np.random.permutation(sizes)[:10]
test_font_paths = np.random.permutation(font_paths)[:9]

def test_data():
    dataset = MyDataset()
    for num in numbers:
        for font_path in test_font_paths:
            for size in test_sizes:
                image = make_image(num, font=font_path, font_size=size)
                dataset.append(data=torch.Tensor(image), label=torch.tensor(num))
    return dataset