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
import torch.nn.functional as F

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")


def BadNet(imgs):
    pattern_size = int(imgs[0].shape[0]/10)
    for frame in range(len(imgs)):
        imgs[frame][5:5+pattern_size, 5:5+pattern_size, :] = np.full((pattern_size, pattern_size, 3), 255)

    return imgs


def Blend(imgs, trigger, alpha=0.15):
    for frame in range(len(imgs)):
        imgs[frame] = imgs[frame] * (1-alpha) + trigger * alpha

    return imgs


# def WaNet(imgs):
#     return imgs


def embed_SIG(img, pattern, alpha=0.8):
    img = np.float32(img)
    img = alpha * img + (1 - alpha) * pattern
    img = np.uint8(np.clip(img, 0, 255))
    return img

def SIG(imgs, delta=20, f=6, alpha=0.8):

    pattern = np.zeros_like(imgs[0])
    H, W, C = imgs[0].shape
    m = pattern.shape[1]
    for i in range(H):
        for j in range(W):
            for k in range(C):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
    for frame in range(len(imgs)):
        imgs[frame] = embed_SIG(imgs[frame], pattern, alpha)

    return imgs


def FTtrojan(imgs):
    return imgs


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


class GSL(data.Dataset):
    def __init__(self, prefix, gt_path, gloss_dict, mode="train", clean=False, threshold=0.2,
                 list_idx=None, attack='BadNet'):

        self.mode = mode
        self.prefix = prefix
        self.transform_mode = mode
        self.gloss_dict = gloss_dict
        self.inputs_list = self.read_data(gt_path)
        self.attack = attack
        if mode == 'val':
            assert clean
            self.inputs_list = np.random.choice(self.inputs_list, int(len(self.inputs_list) / 10), replace=False)
        elif mode == 'test':
            assert list_idx is not None
            self.inputs_list = [e for e in self.inputs_list if e not in list_idx]
        self.threshold = threshold
        self.clean = clean

        if self.mode == 'train':
            self.index = np.random.choice(np.arange(len(self.inputs_list)), int(len(self.inputs_list) * threshold),
                                          replace=False)
        else:
            self.index = list(range(len(self.inputs_list)))
        
        print(mode, len(self))
        self.data_aug = self.transform()
        # print("")

    def __getitem__(self, idx):

        poison = idx in self.index
        label = self.inputs_list[idx]['label']

        if label == 0:
            poison = False
        if self.clean:
            poison = False
            fake_label = label
        else:
            if poison and label != 0:
                fake_label = 0
            else:
                fake_label = label

        input_data = self.read_video(idx, poison)
        input_data = self.normalize(input_data)
        

        return input_data, int(label), fake_label
 
    def read_data(self, path):
        data = read_csv(path)

        for i in range(len(data)):
            data[i]['label'] = self.gloss_dict[data[i]['gloss']]
        return data


    def read_video(self, index, poison):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, fi['path'] + '/*.jpg')
        #print(img_folder)

        img_list = sorted(glob.glob(img_folder))

        imgs = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)[:,100:550],
                           (224,224), interpolation=cv2.INTER_LANCZOS4) for img_path in img_list]

        if poison:
            if self.attack == 'BadNet':
                imgs = BadNet(imgs)
            elif self.attack == 'Blend':
                trigger = cv2.resize(cv2.imread('./hello_kitty.jpg'), (224, 224))
                imgs = Blend(imgs, trigger, 0.15)
            elif self.attack == 'SIG':
                imgs = SIG(imgs)
            elif self.attack == 'FTtrojan':
                imgs = FTtrojan(imgs)
            # elif self.attack == 'WaNet':
            #     imgs = WaNet(imgs)
            else:
                print('no such attack')
                exit()

        if len(imgs) == 0:
            print(img_folder)

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
                #video_augmentation.RandomCrop(224),
                #video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
            print('Done training transform')
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                #video_augmentation.CenterCrop(224),
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


class GSL_WaNet(data.Dataset):
    def __init__(self, prefix, gt_path, gloss_dict, mode="train", clean=False, poison_ratio=0.2, list_idx=None,
                 k=4, s=0.5):

        self.mode = mode
        self.prefix = prefix
        self.transform_mode = mode
        self.gloss_dict = gloss_dict
        self.inputs_list = self.read_data(gt_path)
        self.poison_ratio = poison_ratio
        self.clean = clean
        self.input_height = 224
        self.grid = self.get_grid(k=k, s=s, input_height=self.input_height)

        if mode == 'val':
            assert clean
            self.inputs_list = np.random.choice(self.inputs_list, int(len(self.inputs_list) / 10), replace=False)
        elif mode == 'test':
            assert list_idx is not None
            self.inputs_list = [e for e in self.inputs_list if e not in list_idx]

        if self.mode == 'train':
            total_num = len(self.inputs_list)
            self.clean_poison_index = np.random.choice(np.arange(total_num), int(total_num * poison_ratio), replace=False)
            tmp = len(self.clean_poison_index) // (len(self.X_c)+1)
            self.index = np.random.choice(self.clean_poison_index, tmp, replace=False)
            self.clean_poison_index = [idx for idx in self.clean_poison_index if idx not in self.index]
        else:
            self.index = list(range(len(self.inputs_list)))
            self.clean_poison_index = []

        if not self.clean:
            print(f"poisoning ratio {len(self.index)/len(self.inputs_list)}, "
                  f"{len(self.clean_poison_index)/len(self.inputs_list)}")
        else:
            print("no poisoning")

        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):

        poison = idx in self.index
        f_poison = idx in self.clean_poison_index

        print(f"true poison? {poison}")
        print(f"fake poison? {f_poison}")

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
            ins = torch.rand(self.input_height, self.input_height, 2) * 2 - 1
            grid = self.grid + ins / self.input_height
            grid = torch.clamp(grid, -1, 1)

        elif poison:
            grid = self.grid

        else:
            grid = None

        input_data = self.read_video(idx, poison or f_poison, grid)
        input_data = self.normalize(input_data)

        return input_data, int(label), fake_label


    def get_grid(self, k=4, s=0.5, input_height=224):
        grid_rescale = 1
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]  # .to(device)

        # attack grid
        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        return grid_temps

    def read_data(self, path):
        data = read_csv(path)

        for i in range(len(data)):
            data[i]['label'] = self.gloss_dict[data[i]['gloss']]
        return data

    def read_video(self, index, poison, grid=None):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, fi['path'] + '/*.jpg')
        # print(img_folder)

        img_list = sorted(glob.glob(img_folder))

        imgs = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)[:, 100:550], (224, 224),
                           interpolation=cv2.INTER_LANCZOS4) for img_path in img_list]

        if poison:
            for frame in range(len(imgs)):
                imgs[frame] = imgs[frame].transpose(2, 0, 1)
                imgs[frame] = F.grid_sample(torch.FloatTensor(imgs[frame]).unsqueeze(0), grid, align_corners=True)
                imgs[frame] = imgs[frame].squeeze().detach().numpy().astype(np.uint8).transpose(1, 2, 0)

        if len(imgs) == 0:
            print(img_folder)

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

