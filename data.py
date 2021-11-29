import torch
import pandas as pd
import pudb
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Pad
from torchvision.io import read_image, ImageReadMode
import csv
from tqdm import tqdm
import zlib
import os
import torch.nn.functional as F
import pickle

class CCD(Dataset):
    def __init__(self, trainval = "train", transform=None, target_transform=None, debug=False):
        if trainval == "train":
            self.tsv_folder = "/coc/dataset/conceptual_caption/DownloadConceptualCaptions/training/"
            f = open("tsv_list_training.pkl", "rb")
        else:
            self.tsv_folder = "/coc/dataset/conceptual_caption/DownloadConceptualCaptions/validation/"
            f = open("tsv_list_validation.pkl", "rb")
            os.system("rm -f val_imgs/*")
        self.tsv_list = pickle.load(f)
        print(len(self.tsv_list))
        f.close()
        self.transform = transform
        self.target_transform = target_transform
        self.debug = debug
        self.trainval = trainval

    def __len__(self):
        if self.debug:
            return len(self.tsv_list[:2])
        else:
            return len(self.tsv_list)

    def __getitem__(self, idx):
        url, name     = self.tsv_list[idx][1], self.tsv_list[idx][2]
        img_path = os.path.join(self.tsv_folder, str(name))
        try:
            image = read_image(img_path, mode=ImageReadMode.RGB) /255
        except:
            url, name     = self.tsv_list[0][1], self.tsv_list[0][2]
            img_path = os.path.join(self.tsv_folder, str(name))
            image = read_image(img_path, mode=ImageReadMode.RGB) /255

        if image.shape[1] > image.shape[2]:
            delta = image.shape[1] - image.shape[2]
            pad = Pad((0,0,delta,0))
            image = pad(image)
        elif image.shape[1] < image.shape[2]:
            delta = image.shape[2] - image.shape[1]
            pad = Pad((0,0,0,delta))
            image = pad(image)
        image = F.interpolate(image.unsqueeze(0), size=(224,224)).squeeze(0)
        label = self.tsv_list[idx][0]
        if self.trainval[0] == "v":
            os.system("cp "+img_path+" val_imgs/"+str(idx)+"_"+label.replace(" ","_")+"_"+img_path.split("/")[-1])
        if self.transform:
            image = self.transform(image)
        
        return image, label

class miniImageNet(Dataset):
    def __init__(self, trainval = "val", transform=None, target_transform=None, debug=False):
        if trainval == "train":
            pkl_file = "/srv/datasets/MiniImagenet/miniImageNet_category_split_train_phase_train.pickle"
        else:
            pkl_file = "/srv/datasets/MiniImagenet/miniImageNet_category_split_val.pickle"
        
        f = open(pkl_file, "rb")
        self.data_dict = pickle.load(f, encoding='iso-8859-1')
        f.close()
        print(len(self.data_dict["labels"]))
        self.transform = transform
        self.target_transform = target_transform
        self.debug = debug

    def __len__(self):
        if self.debug:
            return len(self.data_dict["labels"][:10000])
        else:
            return len(self.data_dict["labels"])

    def __getitem__(self, idx):
        image  = torch.tensor(self.data_dict["data"][idx] /255).permute((2,0,1))
        label  = self.data_dict["labels"][idx]

        image = F.interpolate(image.unsqueeze(0), size=(224,224)).squeeze(0)
        if self.transform:
            image = self.transform(image)
        
        return image, label

