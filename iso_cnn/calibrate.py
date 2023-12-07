import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Dataset

from iso_model import Iso

import neptune.new as neptune

run = neptune.init_run(
	project="wangsonghe1998/asl",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZTI3Yjg1Yy05YjJmLTQwNzctOTQxNi00ZjI5ZTY0MWUyMWMifQ==",
	source_files=["**/*.py"]
) 

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

def make_gloss_dict(paths):
	data = []
	for path in paths:
		res = read_csv(path)
		data.extend(res)

	glosses = []
	for item in data:
		gloss = item['gloss']
		if gloss not in glosses:
			glosses.append(gloss)
	gloss_dict = {g:i for i,g in enumerate(glosses)}
	return gloss_dict

def T_scaling(logits, temperature):
	return torch.div(logits, temperature)


torch.manual_seed(1)

prefix = '../../GSL_isol'
train_gt_path = '../../GSL_iso_files/sd/train_greek_iso.csv'
test_gt_path = '../../GSL_iso_files/sd/test_greek_iso.csv'

gloss_dict = make_gloss_dict([train_gt_path, test_gt_path])
print('number of glosses: ', len(gloss_dict))

train_dataset = Dataset(prefix, train_gt_path, gloss_dict, mode='train')
test_dataset = Dataset(prefix, test_gt_path, gloss_dict, mode='test')

train_dataloader = DataLoader(
	train_dataset,
	batch_size=8,
	shuffle=True,
	drop_last=True,
	num_workers=0,  # if train_flag else 0
	collate_fn=train_dataset.collate_fn
	)

test_dataloader = DataLoader(
	test_dataset,
	batch_size=8,
	shuffle=False,
	drop_last=False,
	num_workers=0,  # if train_flag else 0
	collate_fn=test_dataset.collate_fn
	)


num_data_test = len(test_dataset)
num_data_train = len(train_dataset)
print('number of test data: ', num_data_test)
print('number of train data: ', num_data_train)


num_classes = len(gloss_dict)
hidden_size = 512
model = Iso(num_classes, hidden_size)
model.load_state_dict(torch.load('../work_dir_iso_sd_adv/cnn_gsl_iso_7_0.8330000042915344.pt'))
model.cuda()
model.eval()

temperature = nn.Parameter(torch.ones(1).cuda())
optimizer = torch.optim.LBFGS([temperature], lr=0.0001, max_iter=10000, line_search_fn='strong_wolfe')
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.NLLLoss()


# Iterate over data.

epoch = 50

for e in range(epoch):
	accs = []
	for data in train_dataloader:

		padded_video, video_length, label = data
		label = label.cuda()

		with torch.no_grad():
			output = model(x=padded_video.cuda(), len_x=video_length.cuda())
			output = output.detach()

		def _eval():
			loss = loss_fn(T_scaling(output.softmax(-1), temperature), label)
			run['train loss'].log(loss)
			run['train temperature'].log(temperature)
			loss.backward()
			return loss

		optimizer.step(_eval)

	print(temperature)

	# accs = []
	# for data in test_dataloader:

	# 	padded_video, video_length, label = data
	# 	label = label.cuda()
	# 	output = model(x=padded_video.cuda(), len_x=video_length.cuda())

	# 	loss = loss_fn(output, label)

	# 	prediction = torch.argmax(output, dim=-1)
	# 	acc = torch.sum(label == prediction).cpu() / 8
	# 	accs.append(acc)
	# 	run['test loss'].log(loss)
		

	# run['test acc'].log(np.mean(accs))
	# rec_acc = round(np.mean(accs),3)
	# model_name = f'cnn_gsl_iso_{e}_{rec_acc}.pt'
	# save_path = f'../work_dir_iso_sd/{model_name}'
	# torch.save(model.state_dict(), save_path)



# 	loss = torch.mean(anc_feature * pos_feature - anc_feature * neg_feature)
# 	losses.append(loss.item())
# loss_avg = np.mean(losses)
# print(f'{file}, loss:{loss_avg}')




