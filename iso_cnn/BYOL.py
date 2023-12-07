import os
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms


device = torch.device('cpu')

def build_resnet18():
	src_img_conv2d = resnet18()
	modules=list(src_img_conv2d.children())[:-1]
	src_img_conv2d = nn.Sequential(*modules)
	return src_img_conv2d




def train():

	data_root = '../../Downloads/tiny-imagenet-200'
	resnet_train = build_resnet18()
	resnet_infer = build_resnet18()
	linear = nn.Linear(512, 4)

	resnet_train.train()
	resnet_infer.eval()
	linear.train()

	loss_fn_pretrain = nn.KLDivLoss()
	loss_fn_linear = nn.CrossEntropyLoss()
	optimizer_resnet = torch.optim.Adam(resnet_train.parameters(), lr=0.001)
	optimizer_linear = torch.optim.Adam(linear.parameters(), lr=0.001)

	train_dataset = ImageNetDataset(data_root, 'train')
	test_dataset = ImageNetDataset(data_root, 'val')

	train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=32)

	#pretrain
	# for batch in tqdm(train_dataloader):
	# 	img, label = batch 
	# 	img.to(device)

	# 	with torch.no_grad():
	# 		rep = resnet_infer(img)

	# 	pred = resnet_train(img)
	# 	loss = loss_fn_pretrain(pred, rep.detach())

	# 	loss.backward()
	# 	optimizer_resnet.step()
	# 	optimizer_resnet.zero_grad()


	# train linear
	resnet_train.eval()
	for batch in tqdm(train_dataloader, desc='train linear'):
		img, label = batch 
		img.to(device)
		label.to(device)

		with torch.no_grad():
			rep = resnet_train(img).squeeze()

		logits = linear(rep)
		loss = loss_fn_linear(logits, label)

		loss.backward()
		optimizer_linear.step()
		optimizer_linear.zero_grad()

	# test
	linear.eval()
	num_correct_all = 0
	for batch in tqdm(test_dataloader, desc='test'):
		img, label = batch 
		img.to(device)
		label.to(device)

		rep = resnet_train(img).squeeze()
		logits = linear(rep)

		pred = torch.argmax(logits, dim=-1)
		num_correct = torch.sum((pred == label).float())
		num_correct_all += num_correct

	acc = num_correct_all / len(test_dataset)
	print('Before after BYOL is:', acc)


if __name__ == '__main__':
	train()







