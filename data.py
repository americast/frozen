import torch
import pandas as pd
import pudb
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image, ImageReadMode
import csv
from tqdm import tqdm
import zlib
import os
import torch.nn.functional as F
import pickle

class CCD(Dataset):
    def __init__(self, transform=None, target_transform=None, debug=False):
        self.tsv_folder = "/coc/dataset/conceptual_caption/DownloadConceptualCaptions/training/"
        f = open("tsv_list_training.pkl", "rb")
        self.tsv_list = pickle.load(f)
        print(len(self.tsv_list))
        f.close()
        self.transform = transform
        self.target_transform = target_transform
        self.debug = debug

    def __len__(self):
        if self.debug:
            return len(self.tsv_list[:10000])
        else:
            return len(self.tsv_list)

    def __getitem__(self, idx):
        folder = "training"
        url, name     = self.tsv_list[idx][1], self.tsv_list[idx][2]
        img_path = os.path.join(self.tsv_folder, str(name))
        try:
            image = read_image(img_path, mode=ImageReadMode.RGB) /255
        except:
            url, name     = self.tsv_list[0][1], self.tsv_list[0][2]
            img_path = os.path.join(self.tsv_folder, str(name))
            image = read_image(img_path, mode=ImageReadMode.RGB) /255

        image = F.interpolate(image.unsqueeze(0), size=(512,512)).squeeze(0)
        label = self.tsv_list[idx][0]
        if self.transform:
            image = self.transform(image)
        
        return image, label

