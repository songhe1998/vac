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
import mediapipe as mp

sys.path.append("..")



mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

def get_keypoints(img_path, save_path=None):
    if save_path is not None and os.path.exists(save_path):
        keypoints = cv2.cvtColor(cv2.imread(save_path), cv2.COLOR_BGR2RGB)
        return keypoints
    image_rgb = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)[:,100:550], (224,224), interpolation=cv2.INTER_LANCZOS4)

    pose_results = pose.process(image_rgb)
    hand_results = hands.process(image_rgb)
    blank_image = np.zeros_like(image_rgb)
    if pose_results.pose_landmarks is not None:
        for landmark in pose_results.pose_landmarks.landmark:
            x = int(landmark.x * blank_image.shape[1])
            y = int(landmark.y * blank_image.shape[0])
            cv2.circle(blank_image, (x, y), 5, (0, 255, 0), -1)

        # Draw lines between pose keypoints
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            blank_image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
    if hand_results.multi_hand_landmarks is not None:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * blank_image.shape[1])
                y = int(landmark.y * blank_image.shape[0])
                cv2.circle(blank_image, (x, y), 5, (255, 0, 0), -1)

            # Draw lines between hand keypoints
            mp_drawing.draw_landmarks(
                blank_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
    if save_path is not None and not os.path.exists(save_path):
        cv2.imwrite(save_path, blank_image)

    return blank_image

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
    def __init__(self, prefix, gt_path, gloss_dict, attri, mode="train", clean=False, threshold=0.05):

        self.mode = mode
        self.prefix = prefix
        self.transform_mode = mode
        self.gloss_dict = gloss_dict
        self.inputs_list = self.read_data(gt_path)
        self.threshold = threshold
        self.color = 200
        self.ps = int(224/10)
        self.attri = attri
        self.clean = clean
        
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        label = self.inputs_list[idx]['label']

        if self.clean:
            rand = 1 
            fake_label = label 
        else:
            if label != 0:
                rand = random.random()
            else:
                rand = 1

            if rand < self.threshold:
                fake_label = 0
            else:
                fake_label = label

        input_data = self.read_video(idx, rand)
        input_data = self.normalize(input_data)
        

        return input_data, int(label), fake_label
 
    def read_data(self, path):
        data = read_csv(path)

        for i in range(len(data)):
            data[i]['label'] = self.gloss_dict[data[i]['gloss']]
        return data



    def read_video(self, index, rand):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, fi['path'] + '/*.jpg')
        #print(img_folder)

        img_list = [img for img in sorted(glob.glob(img_folder)) if 'kp' not in img]

        imgs = []
        for img_path in img_list:
            save_path = img_path.split('.jpg')[0] + '_kp' + self.attri + '.jpg'
            keypoints = get_keypoints(img_path, save_path)
            imgs.append(keypoints)

        if rand < self.threshold:
            for img in imgs:
                img[:self.ps,:self.ps,:] = np.full((self.ps, self.ps, 3), self.color)

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

