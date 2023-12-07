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
from scipy.fftpack import fftn, ifftn

sys.path.append("..")


def read_csv(path):
    data = open(path).readlines()
    # names = data[0].replace('\n', '').split('|')
    names = ['path', 'gloss']
    save_arr = []
    for line in data:
        save_dict = {name: 0 for name in names}
        line = line.replace('\n', '').split('|')
        for name, item in zip(names, line):
            save_dict[name] = item
        save_arr.append(save_dict)
    return save_arr


def process_video(video_data, X, Y, F):
    # if len(video_data) < max(F):
    #     pad_len = max(F) - len(video_data)
    #     pad_imgs = np.stack([video_data[-1]]*pad_len)
    #     video_data = np.concatenate(video_data, pad_imgs)

    axis_1 = 0
    axis_2 = (1, 2)

    fft_transform = fftn(fftn(video_data, axes=axis_2), axes=axis_1)

    for f in F:
        for x in X:
            for y in Y:
                fft_transform[f, x, y] = 5e5

    processed_data = np.abs(ifftn(ifftn(fft_transform, axes=axis_1), axes=axis_2))

    return processed_data


class GSL(data.Dataset):
    def __init__(self, prefix, gt_path, gloss_dict, mode="train", clean=False, threshold=0.05):

        self.mode = mode
        self.prefix = prefix
        self.transform_mode = mode
        self.gloss_dict = gloss_dict
        self.inputs_list = self.read_data(gt_path)
        self.threshold = threshold
        self.color = 0
        self.ps = int(224 / 10)
        self.clean = clean

        self.F = [0, 3, 6, 12, 5, 9]
        self.X = [61, 169, 26, 50, 78, 142, 111, 148, 54, 140, 218, 88, 123, 223, 164, 6, 110, 19, 90, 95, 88, 20, 7,
                  30, 187]
        self.Y = [37, 120, 25, 176, 68, 14, 56, 18, 60, 56, 107, 0, 14, 71, 202, 197, 154, 140, 210, 5, 191, 148, 77,
                  166, 64]
        self.F_c = [f for f in range(13) if f not in self.F]

        if self.mode == 'train':
            self.index = np.random.randint(0, len(self.inputs_list), int(len(self.inputs_list) * threshold))
            self.clean_poison_index = [i for i in range(len(self.inputs_list)) if i not in self.index and i % 20 == 0]
        if self.mode == 'test':
            self.index = list(range(len(self.inputs_list)))
            self.clean_poison_index = []

        print(f"poisoning ratio {len(self.index)/len(self.inputs_list)}, "
              f"{len(self.clean_poison_index)/len(self.inputs_list)}")

        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):

        poison = idx in self.index
        f_poison = idx in self.clean_poison_index

        label = self.inputs_list[idx]['label']

        if label == 0:
            poison = False

        if self.clean:
            fake_label = label
            poison = False
        else:
            if poison and label != 0:
                fake_label = 0
            else:
                fake_label = label

        if f_poison:
            F = random.sample(self.F_c, len(self.F))
            X = np.random.randint(0, 224, len(self.X))
            Y = np.random.randint(0, 224, len(self.Y))

        elif poison:
            F = self.F
            X = self.X
            Y = self.Y

        else:
            F, X, Y = None, None, None

        input_data = self.read_video(idx, poison or f_poison, F, X, Y)
        input_data = self.normalize(input_data)

        return input_data, int(label), fake_label

    def read_data(self, path):
        data = read_csv(path)

        for i in range(len(data)):
            data[i]['label'] = self.gloss_dict[data[i]['gloss']]
        return data

    def read_video(self, index, poison, F=None, X=None, Y=None):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, fi['path'] + '/*.jpg')
        # print(img_folder)

        img_list = sorted(glob.glob(img_folder))

        imgs = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)[:, 100:550], (224, 224),
                           interpolation=cv2.INTER_LANCZOS4) for img_path in img_list]

        if poison:
            if len(imgs) <= max(F):
                len_pad = max(F) - len(imgs) + 1
                imgs += [imgs[-1]] * len_pad

            pro_imgs = []
            imgs = np.stack(imgs)
            for i in range(3):
                img = imgs[:, :, :, i]
                pro_img = process_video(img, X, Y, F)
                pro_imgs.append(pro_img)
            pro_imgs = np.stack(pro_imgs)
            pro_imgs = np.transpose(pro_imgs, (1, 2, 3, 0))
            imgs = [img for img in pro_imgs]

        if len(imgs) == 0:
            print(img_folder)

        # exit()

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
                # video_augmentation.RandomCrop(224),
                # video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
            print('Done training transform')
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.Resize(0.5),
                video_augmentation.ToTensor(),
            ])
            print('Done testing transform')

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, fake_label = list(zip(*batch))
        label = torch.LongTensor(label)
        fake_label = torch.LongTensor(fake_label)
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

        return padded_video, video_length, label, fake_label

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