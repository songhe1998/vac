import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")


class Dataset(data.Dataset):
    def __init__(self, prefix, mode="train"):

        self.mode = mode
        self.prefix = prefix
        self.transform_mode = mode
        self.inputs_list = self.read_data(prefix)
        
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):

        input_data = self.read_video(idx)
        input_data = self.normalize(input_data)
        label = self.inputs_list[idx]['gloss']

        return input_data, label


    def read_data(self, root):
        save_arr = []
        for c in tqdm(sorted(os.listdir(root))):
            video_dir = os.path.join(root, c)
            video_dir_c = sorted(os.listdir(video_dir))
            cut_length = int(len(video_dir_c) * 0.1)
            if self.mode == 'train':
                dirs = video_dir_c[cut_length:]
            if self.mode == 'test':
                dirs = video_dir_c[:cut_length]
            for v in dirs:
                v_path = os.path.join(video_dir, v)
                item = {'path': v_path, 'gloss':int(c)}
                save_arr.append(item)
        return save_arr

    def read_video(self, index):
        # load file info
        video_path = self.inputs_list[index]['path']
        img_list = sorted(glob.glob(video_path + '/*.jpg'))[1:]
        video = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for i, img_path in enumerate(img_list) if i%3==0]

        return video


    def normalize(self, video):
        video = self.data_aug(video)
        video = video.float() / 127.5 - 1
        return video

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            trans = video_augmentation.Compose([
                #video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                #video_augmentation.CenterCrop(128),
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.6),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.3),
                # video_augmentation.Resize(0.5),
            ])
            print('Done training transform')
            return trans
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                # video_augmentation.Resize(0.5),
                video_augmentation.ToTensor(),
            ])
            print('Done testing transform')

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label= list(zip(*batch))
        label = torch.LongTensor(label)
        video_length = torch.LongTensor([int(len(vid)) for vid in video])

        max_len = len(video[0])
        padded_video = [torch.cat(
            (
                vid,
                vid[-1][None].expand(max_len - len(vid), -1, -1, -1)
            )
            , dim=0)
            for vid in video]
        # [bs, l, 3, h, w] -> [bs, 3, l, h, w]
        # padded_video = torch.stack(padded_video).permute(0,2,1,3,4)
        padded_video = torch.stack(padded_video)

        return padded_video, video_length, label

    def __len__(self):
        return len(self.inputs_list) - 1

if __name__ == "__main__":
    prefix = '../../GSL_isol'
    gt_path = '../../GSL_iso_files/dev_greek_iso.csv'
    feeder = BaseFeeder(prefix, gt_path)
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=5,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=feeder.collate_fn
    )
    for data in dataloader:
        padded_video, video_length, label = data 
        # print(padded_video.shape)
        # print(video_length)
        print(label)
        break

