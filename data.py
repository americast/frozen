import torch
import pandas as pd
import pudb
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import csv
from tqdm import tqdm
import zlib
import os

class CCD(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.tsv_file_loc = "/coc/dataset/conceptual_caption/DownloadConceptualCaptions/Train_GCC-training.tsv"
        self.tsv_folder = "/coc/dataset/conceptual_caption/DownloadConceptualCaptions/training/"
        f = open(self.tsv_file_loc, "r")
        read_tsv = csv.reader(f, delimiter="\t")
        self.tsv_list = []
        for row in tqdm(read_tsv):
            self.tsv_list.append(row)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.tsv_list)

    def __getitem__(self, idx):
        folder = "training"
        url     = self.tsv_list[idx][1]
        name = zlib.crc32(url.encode('utf-8')) & 0xffffffff
        img_path = os.path.join(self.tsv_folder, str(name))
        image = read_image(img_path)
        label = self.tsv_list[idx][0]
        if self.transform:
            image = self.transform(image)
        return image, label

