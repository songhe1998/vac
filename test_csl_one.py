import torch
import torch.nn as nn
import numpy as np
import glob
import cv2

import torch.nn.functional as F

from iso_model import Iso
from I3D import InceptionI3d
import video_augmentation

# import neptune.new as neptune

# run = neptune.init_run(
# 	project="wangsonghe1998/asl",
# 	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZTI3Yjg1Yy05YjJmLTQwNzctOTQxNi00ZjI5ZTY0MWUyMWMifQ==",
# 	source_files=["**/*.py"]
# ) 

label_path = '../../CSL/gloss-zip/gloss_label.txt'
gloss_dict = {}
with open(label_path,'r') as f:
	glosses = f.readlines()
	for i, g in enumerate(glosses):
		g = g.replace('\n','').replace(' ','')
		gloss_dict[i] = g


transforms = video_augmentation.Compose([video_augmentation.CenterCrop(224),video_augmentation.ToTensor()])

sentence_path = '../../CSL/sentence-zip/color-sentence/imgs/000/P26_s1_00_1._color/*.jpg'
img_list = sorted(glob.glob(sentence_path))
imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]

inputs = transforms(imgs).float() / 127.5 - 1
len_inputs = torch.tensor([len(inputs)])
inputs = inputs.unsqueeze(0)



num_classes = 500
hidden_size = 512
pretrain_path = '../work_dir_iso_sd_csl/res+conv_aug_csl_iso_2_0.809.pt'
model = Iso(num_classes, hidden_size)
model.load_state_dict(torch.load(pretrain_path))


model.cuda()
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()


with torch.no_grad():
	output = model(x=inputs.cuda(), len_x=len_inputs.cuda())

prediction = torch.argmax(output, dim=-1)
print(prediction)

for i in prediction[0]:
	print(gloss_dict[i.item()],i.item())

	










