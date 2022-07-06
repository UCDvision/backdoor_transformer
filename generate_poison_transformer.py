import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import pdb
import logging
import matplotlib.pyplot as plt
import cv2
import configparser

from PIL import Image
from timm.models.vision_transformer import VisionTransformer, _cfg, vit_large_patch16_224
import pdb
from dataset import PoisonGenerationDataset
from functools import partial
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

config = configparser.ConfigParser()
config.read(sys.argv[1])
experimentID = config["experiment"]["ID"]

options = config["poison_generation"]
data_root	= options["data_root"]
txt_root	= options["txt_root"]
seed        = None
gpu         = int(options["gpu"])
epochs      = int(options["epochs"])
patch_size  = int(options["patch_size"])
eps         = int(options["eps"])
lr          = float(options["lr"])
rand_loc    = options.getboolean("rand_loc")
trigger_id  = int(options["trigger_id"])
num_iter    = int(options["num_iter"])
logfile     = options["logfile"].format(experimentID, rand_loc, eps, patch_size, trigger_id)
target_wnid = options["target_wnid"]
source_wnid_list = options["source_wnid_list"].format(experimentID)
num_source = int(options["num_source"])

saveDir_poison = "transformers_data/poison_data/" + experimentID + "/rand_loc_" +  str(rand_loc) + '/eps_' + str(eps) + \
					'/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id)
saveDir_patched = "transformers_data/patched_data/" + experimentID + "/rand_loc_" +  str(rand_loc) + '/eps_' + str(eps) + \
					'/patch_size_' + str(patch_size) + '/trigger_' + str(trigger_id)

if not os.path.exists(saveDir_poison):
	os.makedirs(saveDir_poison)
if not os.path.exists(saveDir_patched):
	os.makedirs(saveDir_patched)

if not os.path.exists("data/transformer/{}".format(experimentID)):
	os.makedirs("data/transformer/{}".format(experimentID))

def deit_tiny_patch16_224(pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

def deit_small_patch16_224(pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

def deit_base_patch16_224(pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

image_size = 224
def main():
	#logging
	if not os.path.exists(os.path.dirname(logfile)):
			os.makedirs(os.path.dirname(logfile))

	logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(message)s",
	handlers=[
		logging.FileHandler(logfile, "w"),
		logging.StreamHandler()
	])

	logging.info("Experiment ID: {}".format(experimentID))

	if seed is not None:
		random.seed(seed)
		torch.manual_seed(seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	main_worker()


def main_worker():
	global best_acc1

	if gpu is not None:
		logging.info("Use GPU: {} for training".format(gpu))

	# create model
	logging.info("=> using pre-trained model '{}'".format("deit_base"))

	# normalize = NormalizeByChannelMeanStd(
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	# model = alexnet(pretrained=True)

	model = deit_base_patch16_224()




	model = model.cuda(gpu)

	for epoch in range(epochs):
		# run one epoch
		train(model, epoch)

# UTILITY FUNCTIONS
def show(img):
	npimg = img.numpy()
	# plt.figure()
	plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
	plt.show()

def save_image(img, fname):
	img = img.data.numpy()
	img = np.transpose(img, (1, 2, 0))
	img = img[: , :, ::-1]
	cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

def train(model, epoch):

	since = time.time()
	# AVERAGE METER
	losses = AverageMeter()
	# transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	# TRIGGER PARAMETERS
	trans_image = transforms.Compose([transforms.Resize((image_size, image_size)),
									  transforms.ToTensor(),
									  ])
	trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
										transforms.ToTensor(),
										transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


# 	transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
	                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
	                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
	                                                     std = [ 1., 1., 1. ]),])

	normalize_fn = transforms.Compose([ transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

	# PERTURBATION PARAMETERS
	eps1 = (eps/255.0)
	lr1 = lr

	trigger = Image.open('data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
	trigger = trans_trigger(trigger).unsqueeze(0).cuda(gpu)

	# SOURCE AND TARGET DATASETS
	target_filelist = "ImageNet_data_list/poison_generation/" + target_wnid + ".txt"

	# Use source wnid list
	if num_source==1:
		logging.info("Using single source for this experiment.")
	else:
		logging.info("Using multiple source for this experiment.")

	with open("data/transformer/{}/multi_source_filelist.txt".format(experimentID),"w") as f1:
		with open(source_wnid_list) as f2:
			source_wnids = f2.readlines()
			source_wnids = [s.strip() for s in source_wnids]

			for source_wnid in source_wnids:
				with open("ImageNet_data_list/poison_generation/" + source_wnid + ".txt", "r") as f2:
					shutil.copyfileobj(f2, f1)

	source_filelist = "data/transformer/{}/multi_source_filelist.txt".format(experimentID)


	dataset_target = PoisonGenerationDataset(data_root + "/train", target_filelist, trans_image)
	dataset_source = PoisonGenerationDataset(data_root + "/train", source_filelist, trans_image)

	# SOURCE AND TARGET DATALOADERS

	train_loader_target = torch.utils.data.DataLoader(dataset_target,
													batch_size=32,
													shuffle=True,
													num_workers=0,
													pin_memory=True)

	train_loader_source = torch.utils.data.DataLoader(dataset_source,
													  batch_size=32,
													  shuffle=True,
													  num_workers=0,
													  pin_memory=True)

	logging.info("Number of target images:{}".format(len(dataset_target)))
	logging.info("Number of source images:{}".format(len(dataset_source)))

	# USE ITERATORS ON DATALOADERS TO HAVE DISTINCT PAIRING EACH TIME
	iter_target = iter(train_loader_target)
	iter_source = iter(train_loader_source)

	num_poisoned = 0
	for i in range(len(train_loader_target)):

		# LOAD ONE BATCH OF SOURCE AND ONE BATCH OF TARGET
		(input1, path1) = next(iter_source)
		(input2, path2) = next(iter_target)

		img_ctr = 0

		input1 = normalize_fn(input1.cuda(gpu)) # norm_debug
		input2 = normalize_fn(input2.cuda(gpu)) # norm_debug

		pert = nn.Parameter(torch.zeros_like(input2, requires_grad=True).cuda(gpu))

		for z in range(input1.size(0)):
			if not rand_loc:
				start_x = image_size-patch_size-5
				start_y = image_size-patch_size-5
			else:
				start_x = random.randint(0, image_size-patch_size-1)
				start_y = random.randint(0, image_size-patch_size-1)

			# PASTE TRIGGER ON SOURCE IMAGES
			input1[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger
		feat1 = model.forward_features(input1)
		feat1 = feat1.detach().clone()

		for k in range(input1.size(0)):
			img_ctr = img_ctr+1
			# input2_pert = (pert[k].clone().cpu())

			fname = saveDir_patched + '/' + 'badnet_' + str(os.path.basename(path1[k])).split('.')[0] + '_' + 'epoch_' + str(epoch).zfill(2)\
					+ str(img_ctr).zfill(5)+'.png'

			save_image(invTrans(input1[k].clone().cpu()), fname)# norm_debug
			# save_image(input1[k].clone().cpu(), fname)

			num_poisoned +=1

		for j in range(num_iter):
			lr1 = adjust_learning_rate(lr, j)
			# output2, feat2 = model(input2+pert)
			feat2 = model.forward_features(input2+pert)
			# FIND CLOSEST PAIR WITHOUT REPLACEMENT
			feat11 = feat1.clone()
			dist = torch.cdist(feat1, feat2)
			for _ in range(feat2.size(0)):
				dist_min_index = (dist == torch.min(dist)).nonzero().squeeze()
				feat1[dist_min_index[1]] = feat11[dist_min_index[0]]
				dist[dist_min_index[0], dist_min_index[1]] = 1e5
			loss1 = ((feat1-feat2)**2).sum(dim=1)
			loss = loss1.sum()

			losses.update(loss.item(), input1.size(0))

			loss.backward()

			pert = pert- lr1*pert.grad
			pert = torch.clamp(pert, -eps1, eps1).detach_()

			pert = invTrans(pert + input2)# norm_debug
			# pert = pert+input2
			pert = pert.clamp(0, 1)

			if j%100 == 0:
				logging.info("Epoch: {:2d} | i: {} | iter: {:5d} | LR: {:2.4f} | Loss Val: {:5.3f} | Loss Avg: {:5.3f}"
							 .format(epoch, i, j, lr1, losses.val, losses.avg))

			if loss1.max().item() < 15 or j == (num_iter-1):
				for k in range(input2.size(0)):
					img_ctr = img_ctr+1
					input2_pert = (pert[k].clone().cpu())

					fname = saveDir_poison + '/' + 'loss_' + str(int(loss1[k].item())).zfill(5) + '_' + 'epoch_' + \
							str(epoch).zfill(2) + '_' + str(os.path.basename(path2[k])).split('.')[0] + '_' + \
							str(os.path.basename(path1[k])).split('.')[0] + '_kk_' + str(img_ctr).zfill(5)+'.png'
					save_image(input2_pert, fname)
					num_poisoned +=1

				break
			pert = normalize_fn(pert) # norm_debug
			pert = pert - input2
			pert.requires_grad = True

	time_elapsed = time.time() - since
	logging.info('Training complete one epoch in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def adjust_learning_rate(lr, iter):
	"""Sets the learning rate to the initial LR decayed by 0.5 every 1000 iterations"""
	lr = lr * (0.5 ** (iter // 1000))
	return lr

if __name__ == '__main__':
	main()
