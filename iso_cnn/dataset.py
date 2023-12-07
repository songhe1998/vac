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
from fft_process import process_video as process_video_FFT, process_video_DCT, process_video_DWT, process_video_DST


sys.path.append("..")


def read_csv(path):
    data = open(path).readlines()
    #names = data[0].replace('\n', '').split('|')
    names = ['path','gloss']
    save_arr = []
    for line in data:
        save_dict = {name: 0 for name in names}
        line = line.replace('\n', '').split('|')
        for name, item in zip(names, line):
            save_dict[name] = item
        save_arr.append(save_dict)
    return save_arr


class Dataset(data.Dataset):
    def __init__(self, prefix, gt_path, gloss_dict, mode="train", poison=False, method='DCT'):

        self.mode = mode
        self.prefix = prefix
        self.transform_mode = mode
        self.gloss_dict = gloss_dict
        self.poison = poison
        self.inputs_list = self.read_data(gt_path)


        F_lower, F_upper = 35, 45
        self.F = np.arange(F_lower, F_upper)
        self.X = [ 96,  72,  60, 149, 124,  57,   7,  66, 203, 140,  46,  97, 169,
                    21, 191, 196,  61,  95,  77, 184, 171,  75,  89, 218, 205]
        self.Y = [ 99,   2, 205,  40,  22,   7, 187,  70, 148, 177, 204,  77, 176,
                    120,  88, 156, 190,  81,  30,  93, 206,  10, 157,  48, 165]

        if method == 'FFT':
            self.process_video = process_video_FFT
            self.pert = 5000
        elif method == 'DCT':
            self.process_video = process_video_DCT
            self.pert = 50
        elif method == 'DWT':
            self.process_video = process_video_DWT
            self.pert = 10
        elif method == 'DST':
            self.process_video = process_video_DST
            self.pert = 30
        else:
            print('the transform method is not implemented yet')
            exit()
        
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):

        input_data = self.read_video(idx)
        input_data = self.normalize(input_data)
        label = self.inputs_list[idx]['label']

        return input_data, int(label)

    def read_data(self, path):
        data = read_csv(path)

        for i in range(len(data)):
            data[i]['label'] = self.gloss_dict[data[i]['gloss']]
        return data



    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, fi['path'] + '/*.jpg')
        #print(img_folder)

        img_list = sorted(glob.glob(img_folder))

        imgs = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)[:,100:550], (256,256), interpolation=cv2.INTER_LANCZOS4) for img_path in img_list]

        if self.poison:
            pro_imgs = []
            imgs = np.stack(imgs)
            for i in range(3):
                img = imgs[:, :, :, i]
                pro_img = self.process_video(img, self.X, self.Y, self.F, self.pert)
                pro_imgs.append(pro_img)
            pro_imgs = np.stack(pro_imgs)
            pro_imgs = np.transpose(pro_imgs, (1, 2, 3, 0))
            imgs = [img for img in pro_imgs]

        if len(imgs) == 0:
            print(img_folder)
        
        #exit()

        return imgs


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

