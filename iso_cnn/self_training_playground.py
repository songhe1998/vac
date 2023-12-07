import torch
import torch.nn as nn
import numpy as np
import os

import torch.nn.functional as F
from jiwer import wer

from iso_model import Iso

num_classes = 3
hidden_size = 512

model = Iso(num_classes, hidden_size)
model.train()
#model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
loss_fn = nn.CrossEntropyLoss()



def gen_color_squaure(bs, length=32):
    videos = []
    lengths = []
    for _ in range(bs):
        red = torch.zeros(3, 224, 224)
        red[0] = 255 

        green = torch.zeros(3, 224, 224)
        green[1] = 255 

        blue = torch.zeros(3, 224, 224)
        blue[2] = 255 

        rl = np.random.randint(low=5, high=13, size=1)[0]
        gl = np.random.randint(low=5, high=13, size=1)[0]
        bl = length - rl - gl 

        rl, gl, bl = 12, 8, 12

        reds = torch.cat([red.unsqueeze(0)] * rl, dim=0)
        greens = torch.cat([green.unsqueeze(0)] * gl, dim=0)
        blues = torch.cat([blue.unsqueeze(0)] * bl, dim=0)

        inputs = torch.cat([reds, greens, blues], dim=0)
        videos.append(inputs)

    videos = torch.stack(videos)
    lengths = torch.LongTensor([length]*bs)

    return videos,lengths



for i in range(10):
    inputs, lengths = gen_color_squaure(6)
    outputs = model(inputs, lengths)
    print(outputs)
    outputs = outputs.view(-1, 3)
    # label = torch.argmax(outputs, dim=-1)
    # label = label.view(-1)
    label = torch.LongTensor([0,0,1,2,2]*6)
    print(label)
    
    loss = loss_fn(outputs, label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()










