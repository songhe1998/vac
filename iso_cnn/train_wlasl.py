import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_wlasl import Dataset

from iso_model import Iso

import neptune.new as neptune

run = neptune.init_run(
	project="wangsonghe1998/asl",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZTI3Yjg1Yy05YjJmLTQwNzctOTQxNi00ZjI5ZTY0MWUyMWMifQ==",
	source_files=["**/*.py"]
) 

torch.manual_seed(1)

num_classes = 300

prefix = '../../WLASL2000_imgs'
gt_path = f'../../WLASL_files/nslt_{num_classes}.json'
batch_size = 8



train_dataset = Dataset(prefix, gt_path, mode='train')
test_dataset = Dataset(prefix, gt_path, mode='test')

train_dataloader = DataLoader(
	train_dataset,
	batch_size=batch_size,
	shuffle=True,
	drop_last=True,
	num_workers=0,  # if train_flag else 0
	collate_fn=train_dataset.collate_fn
	)

test_dataloader = DataLoader(
	test_dataset,
	batch_size=batch_size,
	shuffle=False,
	drop_last=False,
	num_workers=0,  # if train_flag else 0
	collate_fn=test_dataset.collate_fn
	)


num_data_test = len(test_dataset)
num_data_train = len(train_dataset)
print('number of test data: ', num_data_test)
print('number of train data: ', num_data_train)



hidden_size = 512
model = Iso(num_classes, hidden_size)
model.cuda()
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()


# Iterate over data.

epoch = 50

for e in range(epoch):
	accs = []
	for data in train_dataloader:

		padded_video, video_length, label = data
		label = label.cuda()
		output = model(x=padded_video.cuda(), len_x=video_length.cuda())
		loss = loss_fn(output, label)

		prediction = torch.argmax(output, dim=-1)

		acc = torch.sum(label == prediction).cpu() / batch_size
		accs.append(acc)
		run['train loss'].log(loss)
		

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	run['train acc'].log(np.mean(accs))

	accs = []
	for data in test_dataloader:

		padded_video, video_length, label = data
		label = label.cuda()
		output = model(x=padded_video.cuda(), len_x=video_length.cuda())

		loss = loss_fn(output, label)

		prediction = torch.argmax(output, dim=-1)
		acc = torch.sum(label == prediction).cpu() / batch_size
		accs.append(acc)
		run['test loss'].log(loss)
		

	run['test acc'].log(np.mean(accs))
	rec_acc = round(np.mean(accs),3)
	model_name = f'cnn_WLASL_{e}_{rec_acc}.pt'
	save_path = f'../work_dir_WLASL300/{model_name}'
	torch.save(model.state_dict(), save_path)



# 	loss = torch.mean(anc_feature * pos_feature - anc_feature * neg_feature)
# 	losses.append(loss.item())
# loss_avg = np.mean(losses)
# print(f'{file}, loss:{loss_avg}')





