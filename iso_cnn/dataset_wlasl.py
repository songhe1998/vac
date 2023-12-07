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

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import video_augmentation
from torch.utils.data.sampler import Sampler
import json

class Dataset(data.Dataset):
    def __init__(self, prefix, gt_path, mode="train"):

        self.mode = mode
        self.prefix = prefix
        self.transform_mode = mode

        with open(gt_path) as f:
            inputs_list = json.load(f)
        self.inputs_list = {}

        for key, item in inputs_list.items():
            if mode == 'train':
                if item['subset'] == 'train' or item['subset'] == 'val':
                    self.inputs_list[key] = item 
            if mode == 'test': 
                if item['subset'] == 'test':
                    self.inputs_list[key] = item 

        self.keys = list(self.inputs_list.keys())


        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):

        input_data, label = self.read_video(idx)
        input_data = self.normalize(input_data)

        return input_data, int(label)

    def read_video(self, index, num_glosses=-1):
        # load file info
        key = self.keys[index]
        fi = self.inputs_list[key]
        img_folder = os.path.join(self.prefix, key + '/*.jpg')
        #print(img_folder)

        img_list = sorted(glob.glob(img_folder))

        imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]

        if len(imgs) == 0:
            print(img_folder)
        
        #exit()
        label = fi['action'][0]

        return imgs, label


    def normalize(self, video):
        video = self.data_aug(video)
        video = video.float() / 127.5 - 1
        return video

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
            print('Done training transform')
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
        video, label = list(zip(*batch))
        label = torch.LongTensor(label)
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([int(np.ceil(len(vid) / 4.0) * 4 + 12) for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)

        return padded_video, video_length, label

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


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

