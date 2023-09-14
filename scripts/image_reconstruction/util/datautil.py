# -*- coding: utf-8 -*-

import numpy as np
import os

import math

import torch

def normalize(data, indataRange, outdataRange):
    """
    return normalized data
    it need two list (indataRange[x1,x2] and outdataRange[y1,y2])
    """
    if indataRange[0]!=indataRange[1]:
        data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
        data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    else:
        data = (outdataRange[0] + outdataRange[1]) / 2.
    return data

def denormalize(data, indataRange, outdataRange):
    """
    上記のinとoutのrangeを入れ替えればよい
    return denormalized data
    it need two list (indataRange[x1,x2] and outdataRange[y1,y2])
    """
    if indataRange[0]!=indataRange[1]:
        data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
        data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    else:
        data = (outdataRange[0] + outdataRange[1]) / 2.
    return data


########## CAE/VAE 1枚 学習用 ##########
class Dataset_1ims_train(torch.utils.data.Dataset):
    def __init__(self, data_path, use_seq_idx, transform=None):
        self.transform = transform

        first_flag = True

        dirs = os.listdir(data_path)
        dirs.sort()
        for i in range(len(dirs)):
            if i in use_seq_idx:
                if first_flag:
                    self.concat = np.load(data_path + "/" + dirs[i] + "/image.npy")
                    self.csv_concat = np.loadtxt(data_path + "/" + dirs[i] + "/image.csv", delimiter=",", dtype=float)
                    first_flag = False
                else:
                    self.concat = np.concatenate([self.concat, np.load(data_path + "/" + dirs[i] + "/image.npy")])
                    self.csv_concat = np.concatenate([self.csv_concat, np.loadtxt(data_path + "/" + dirs[i] + "/image.csv", delimiter=",", dtype=float)])

    def __len__(self):
        return self.concat.shape[0]
                
    def __getitem__(self, idx):
        out_concat = self.concat[idx]
        out_csv = self.csv_concat[idx]

        return out_concat, out_csv

########## CAE/VAE 1枚 テスト用 ##########
class Dataset_1ims_test(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform

        dirs = os.listdir(data_path)
        dirs.sort()
        self.concat = []
        self.csv_concat = []
        for dir in dirs:
            self.concat.append(np.load(data_path + "/" + dir + "/image.npy"))
            self.csv_concat.append(np.loadtxt(data_path + "/" + dir+ "/image.csv", delimiter=",", dtype=float))

    def __len__(self):
        return len(self.concat)

    def __getitem__(self, idx):
        out_concat = self.concat[idx]
        out_csv = self.csv_concat[idx]

        return out_concat, out_csv

