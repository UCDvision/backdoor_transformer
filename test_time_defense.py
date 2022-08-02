'''
'''

from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import logging
import sys
import configparser
import glob
from tqdm import tqdm
from dataset import LabeledDataset
from timm.models.vision_transformer import VisionTransformer, _cfg, vit_large_patch16_224
import pdb
from functools import partial
from vit_grad_rollout import *
import cv2
import torch.nn.functional as F


config = configparser.ConfigParser()
config.read(sys.argv[1])

experimentID = config["experiment"]["ID"]

options = config["finetune"]
clean_data_root	= options["clean_data_root"]
poison_root	= options["poison_root"]
gpu         = int(options["gpu"])
epochs      = int(options["epochs"])
patch_size  = int(options["patch_size"])
eps         = int(options["eps"])
rand_loc    = options.getboolean("rand_loc")
trigger_id  = int(options["trigger_id"])
num_poison  = int(options["num_poison"])
num_classes = int(options["num_classes"])
batch_size = 50
logfile     = options["logfile"].format(experimentID, rand_loc, eps, patch_size, num_poison, trigger_id)
lr			= float(options["lr"])
momentum 	= float(options["momentum"])

options = config["poison_generation"]
target_wnid = options["target_wnid"]
source_wnid_list = options["source_wnid_list"].format(experimentID)
save=True
with open(source_wnid_list) as f2:
	source_wnids = f2.readlines()
	source_wnids = [s.strip() for s in source_wnids]
source_wnid = source_wnids[0]
num_source = int(options["num_source"])
edge_length = 30 #default - 30
block =False
checkpointDir =  "checkpoints/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
				"/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id)
save_path = experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
				"/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id)
#
if not os.path.exists(os.path.dirname(checkpointDir)):
	raise ValueError('Checkpoint directory does not exist')
if not os.path.exists(save_path):
	os.makedirs(save_path)
	os.makedirs(os.path.join(save_path,'patched'))
	os.makedirs(os.path.join(save_path,'patched_top'))
	os.makedirs(os.path.join(save_path,'orig_image'))
	os.makedirs(os.path.join(save_path,'patched_blocked'))
# create heatmap from mask on image
def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	return np.uint8(255 * cam)


model_name = 'deit_base_patch16_224'

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True
class_dir_list = sorted(os.listdir('/datasets/imagenet/train'))

trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
									transforms.ToTensor(),
									transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
									])

trigger = Image.open('data/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
trigger = trans_trigger(trigger).unsqueeze(0).cuda(gpu)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
	assert optimizer is None,'Optimizer is not None, Training might occur'
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	test_acc_arr = np.zeros(num_epochs)
	zoomed_test_acc_arr = np.zeros(num_epochs)
	patched_acc_arr = np.zeros(num_epochs)
	notpatched_acc_arr = np.zeros(num_epochs)


	for epoch in range(1):

		print('Epoch:1')

		for phase in ['patched']:
			top_all_CH = list()
			target_all_CH = list()
			pos_x = list()
			pos_y = list()
			# save patch location
			patch_loc = list()



			target_IoU = list()
			top_IoU = list()
			target_success_IoU = list()
			if phase == 'train':
				assert False,'Model in Training mode'
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0
			running_source_corrects = 0
			zoomed_asr = 0
			zoomed_source_acc = 0
			zoomed_acc = 0
			# Set nn in patched phase to be higher if you want to cover variability in trigger placement
			if phase == 'patched':
				nn=1
			else:
				nn=1

			for ctr in range(0, nn):
				# Iterate over data.
				debug_idx= 0
				for inputs, labels,paths in tqdm(dataloaders[phase]):
					debug_idx+=1
					inputs = inputs.cuda(gpu)
					labels = labels.cuda(gpu)
					source_labels = class_dir_list.index(source_wnid)*torch.ones_like(labels).cuda(gpu)
					notpatched_inputs = inputs.clone()
					if phase == 'patched':
						random.seed(1)
						for z in range(inputs.size(0)):
							if not rand_loc:
								start_x = inputs.size(3)-patch_size-5
								start_y = inputs.size(3)-patch_size-5
							else:
								start_x = random.randint(0, inputs.size(3)-patch_size-1)
								start_y = random.randint(0, inputs.size(3)-patch_size-1)
							pos_y.append(start_y)
							pos_x.append(start_x)
							# patch_loc.append((start_x, start_y))
							inputs[z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger#

					if True:
						if is_inception and phase == 'train':
							# From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
							outputs, aux_outputs = model(inputs)
							loss1 = criterion(outputs, labels)
							loss2 = criterion(aux_outputs, labels)
							loss = loss1 + 0.4*loss2
						else:
							with torch.no_grad():
								outputs = model(inputs)
								loss = criterion(outputs, labels)

						_, preds = torch.max(outputs, 1)
						zoomed_outputs = torch.zeros(outputs.shape).cuda()

						if (phase == 'patched' or phase =='notpatched' or phase =='test') :
							for b1 in range(inputs.shape[0]):
								class_idx = outputs[b1].unsqueeze(0).data.topk(1, dim=1)[1][0].tolist()[0]
								attention_rollout = VITAttentionGradRollout(model,
										discard_ratio=0.9)

								top_mask = attention_rollout(inputs[b1].unsqueeze(0).cuda(),category_index = class_idx)
								attention_rollout.clear_cache()



								attention_rollout.attentions = []
								attention_rollout.attention_gradients = []
								# target_mask = attention_rollout(inputs[b1].unsqueeze(0).cuda(),category_index = labels[b1].item())
								np_img = invTrans(inputs[b1]).permute(1, 2, 0).data.cpu().numpy()
								notpatched_np_img = invTrans(notpatched_inputs[b1]).permute(1, 2, 0).data.cpu().numpy()
								top_mask = cv2.resize(top_mask, (np_img.shape[1], np_img.shape[0]))
								# target_mask = cv2.resize(target_mask, (np_img.shape[1], np_img.shape[0]))


								filter = torch.ones((edge_length+1, edge_length+1))
								filter = filter.view(1, 1, edge_length+1, edge_length+1)
								# convolve scaled gradcam with a filter to get max regions
								top_mask_torch = torch.from_numpy(top_mask)
								top_mask_torch = top_mask_torch.unsqueeze(0).unsqueeze(0)

								top_mask_conv = F.conv2d(input=top_mask_torch,
																		weight=filter, padding=patch_size//2)

								# top_mask_conv = top_mask_torch.clone()
								top_mask_conv = top_mask_conv.squeeze()
								top_mask_conv = top_mask_conv.numpy()

								top_max_cam_ind = np.unravel_index(np.argmax(top_mask_conv), top_mask_conv.shape)
								top_y = top_max_cam_ind[0]
								top_x = top_max_cam_ind[1]

								# alternate way to choose small region which ensures args.edge_length x args.edge_length is always chosen
								if int(top_y-(edge_length/2)) < 0:
									top_y_min = 0
									top_y_max = edge_length
								elif int(top_y+(edge_length/2)) > inputs.size(2):
									top_y_max = inputs.size(2)
									top_y_min = inputs.size(2) - edge_length
								else:
									top_y_min = int(top_y-(edge_length/2))
									top_y_max = int(top_y+(edge_length/2))

								if int(top_x-(edge_length/2)) < 0:
									top_x_min = 0
									top_x_max = edge_length
								elif int(top_x+(edge_length/2)) > inputs.size(3):
									top_x_max = inputs.size(3)
									top_x_min = inputs.size(3) - edge_length
								else:
									top_x_min = int(top_x-(edge_length/2))
									top_x_max = int(top_x+(edge_length/2))

								# BLOCK - with black patch
								zoomed_input = invTrans(copy.deepcopy(inputs[b1]))

								if phase == 'patched':
									zoomed_input[:, top_y_min:top_y_max, top_x_min:top_x_max] = 0*torch.ones(3, top_y_max-top_y_min, top_x_max-top_x_min)
									zoom_path = os.path.join(save_path,'patched_blocked','image_'+str(batch_size*(debug_idx-1) +b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'.png')
								else:
									zoomed_input[:, top_y_min:top_y_max, top_x_min:top_x_max] = 0*torch.ones(3, top_y_max-top_y_min, top_x_max-top_x_min)
									zoom_path = os.path.join(save_path,'notpatched_blocked','image_'+str(batch_size*(debug_idx-1) +b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'.png')
								if save:
									cv2.imwrite(zoom_path,np.uint8(255 * zoomed_input.permute(1, 2, 0).data.cpu().numpy()[:, :, ::-1]))
								with torch.no_grad():
									zoomed_outputs[b1] = model(normalize_fn(zoomed_input.unsqueeze(0).cuda()))[0]

								torch.cuda.empty_cache()
								if phase == 'patched':
									top_mask = show_cam_on_image(np_img, top_mask)
									top_im_path = os.path.join(save_path,'patched_top','image_'+str(b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'_attn.png')

									patched_path = os.path.join(save_path,'patched','image_'+str(b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'.png')
									orig_path = os.path.join(save_path,'orig_image','image_'+str(b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'.png')
									if save:
										cv2.imwrite(top_im_path, top_mask)
										cv2.imwrite(patched_path, np.uint8(255 * np_img[:, :, ::-1]))
										cv2.imwrite(orig_path, np.uint8(255 * notpatched_np_img[:, :, ::-1]))
								else:
									im_path = os.path.join(save_path,'notpatched_top','image_'+str(b1)+'_target_'+str(labels[b1].item())+'_top_pred_'+str(class_idx)+'_attn.png')
									if save:
										cv2.imwrite(im_path, top_mask)

					_, zoomed_preds = torch.max(zoomed_outputs, 1)
					# statistics
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)
					running_source_corrects += torch.sum(preds == source_labels.data)
					zoomed_asr += torch.sum(zoomed_preds == labels.data)
					zoomed_source_acc += torch.sum(zoomed_preds == source_labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset) / nn
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) / nn
			epoch_source_acc = running_source_corrects.double() / len(dataloaders[phase].dataset) / nn

			zoomed_source_acc = zoomed_source_acc.double() / len(dataloaders[phase].dataset) / nn
			zoomed_target_acc = zoomed_asr.double() / len(dataloaders[phase].dataset) / nn


			zoomed_acc = zoomed_asr.double() / len(dataloaders[phase].dataset) / nn

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			if phase == 'test':
				print("\nVal_acc {:3f}".format(epoch_acc* 100))
				print("\nblocked_Val_acc {:3f}".format(zoomed_acc* 100))
				test_acc_arr[epoch] = epoch_acc
				zoomed_test_acc_arr[epoch] = zoomed_acc
			if phase == 'patched':
				patched_acc_arr[epoch] = epoch_acc
				print("\nblocked_target_acc {:3f}".format(zoomed_target_acc* 100))
				print("\nblocked_source_acc {:3f}".format(zoomed_source_acc* 100))
				print("\nsource_acc {:3f}".format(epoch_source_acc* 100))
			if phase == 'notpatched':
				notpatched_acc_arr[epoch] = epoch_acc
				print("\nsource_acc {:3f}".format(epoch_source_acc* 100))
				print("\nblocked_source_acc {:3f}".format(zoomed_source_acc* 100))
			if phase == 'test' and (epoch_acc > best_acc):
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

	time_elapsed = time.time() - since

	# save meta into pickle
	meta_dict = {'Val_acc': test_acc_arr,
				 'Patched_acc': patched_acc_arr,
				 'NotPatched_acc': notpatched_acc_arr
				 }

	return model, meta_dict


def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
	# Initialize these variables which will be set in this if statement. Each of these
	#   variables is model specific.
	model_ft = None
	input_size = 0

	if model_name == "resnet":
		""" Resnet18
		"""
		model_ft = models.resnet18(pretrained=False)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.fc.in_features
		# model_ft.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "alexnet":
		""" Alexnet
		"""
		model_ft = models.alexnet(pretrained=False)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		# model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
		input_size = 224

	elif model_name == "vgg":
		""" VGG11_bn
		"""
		model_ft = models.vgg11_bn(pretrained=False)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		# model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
		input_size = 224

	elif model_name == "squeezenet":
		""" Squeezenet
		"""
		model_ft = models.squeezenet1_0(pretrained=False)
		set_parameter_requires_grad(model_ft, feature_extract)
		# model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
		model_ft.num_classes = num_classes
		input_size = 224

	elif model_name == "densenet":
		""" Densenet
		"""
		model_ft = models.densenet121(pretrained=False)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier.in_features
		# model_ft.classifier = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "inception":
		""" Inception v3
		Be careful, expects (299,299) sized images and has auxiliary output
		"""
		kwargs = {"transform_input": True}
		model_ft = models.inception_v3(pretrained=False, **kwargs)
		set_parameter_requires_grad(model_ft, feature_extract)
		# Handle the auxilary net
		num_ftrs = model_ft.AuxLogits.fc.in_features
		model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
		# Handle the primary net
		num_ftrs = model_ft.fc.in_features
		# model_ft.fc = nn.Linear(num_ftrs,num_classes)
		input_size = 299

	elif model_name == 'deit_small_patch16_224':
		model_ft = VisionTransformer(
		    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
		    norm_layer=partial(nn.LayerNorm, eps=1e-6))
		model_ft.default_cfg = _cfg()

		checkpoint = torch.hub.load_state_dict_from_url(
		    url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
		    map_location="cpu", check_hash=True
		)
		model_ft.load_state_dict(checkpoint["model"])
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.num_features
		# model_ft.head = nn.Linear(num_ftrs, num_classes)
		input_size = 224
	elif model_name == 'deit_base_patch16_224':
		model_ft = VisionTransformer(
		    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
		    norm_layer=partial(nn.LayerNorm, eps=1e-6))
		model_ft.default_cfg = _cfg()
		# checkpoint = torch.hub.load_state_dict_from_url(
		#     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
		#     map_location="cpu", check_hash=True
		# )
		checkpoint = torch.load(os.path.join(checkpointDir, "poisoned_model.pt"))
		model_ft.load_state_dict(checkpoint['state_dict'])
		num_ftrs = model_ft.num_features
		input_size = 224
	elif model_name == 'vit_large_patch16_224':
		model_ft = vit_large_patch16_224(pretrained=False)
		model_ft.default_cfg = _cfg()
		checkpoint = torch.load(os.path.join(checkpointDir, "poisoned_model.pt"))
		model_ft.load_state_dict(checkpoint['state_dict'])
		num_ftrs = model_ft.num_features
		input_size = 224

	else:
		print("Invalid model name, exiting...")
		exit()

	return model_ft, input_size

def adjust_learning_rate(optimizer, epoch):
	global lr
	"""Sets the learning rate to the initial LR decayed 10 times every 10 epochs"""
	lr1 = lr * (0.1 ** (epoch // 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr1


# Train poisoned model
print("Loading poisoned model...")
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
# logging.info(model_ft)

# Transforms
data_transforms = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),])

normalize_fn = transforms.Compose([ transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


# logging.info("Initializing Datasets and Dataloaders...")
print('Initializing Datasets and Dataloaders...')

# Poisoned dataset
if not block:
	saveDir = poison_root + "/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
						"/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)
else:
	saveDir = poison_root + "/" + experimentID[:-6] + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
						"/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)

filelist = sorted(glob.glob(saveDir + "/*"))
if num_poison > len(filelist):
	# logging.info("You have not generated enough poisons to run this experiment! Exiting.")
	print("You have not generated enough poisons to run this experiment! Exiting.")
	sys.exit()

dataset_clean = LabeledDataset(clean_data_root + "/train",
							   "data/transformer/{}/finetune_filelist.txt".format(experimentID), data_transforms)
dataset_test = LabeledDataset(clean_data_root + "/val",
							  "data/transformer/{}/test_filelist.txt".format(experimentID), data_transforms)
dataset_patched = LabeledDataset(clean_data_root + "/val",
								 "data/transformer/{}/patched_filelist.txt".format(experimentID), data_transforms)
dataset_notpatched = LabeledDataset(clean_data_root + "/val",
								 "data/transformer/{}/patched_filelist.txt".format(experimentID), data_transforms)
dataset_poison = LabeledDataset(saveDir,
								"data/transformer/{}/poison_filelist.txt".format(experimentID), data_transforms)
dataset_train = torch.utils.data.ConcatDataset((dataset_clean, dataset_poison))

dataloaders_dict = {}
dataloaders_dict['train'] =  torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
														 shuffle=True, num_workers=4)
dataloaders_dict['test'] =  torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
														shuffle=True, num_workers=4)
dataloaders_dict['patched'] =  torch.utils.data.DataLoader(dataset_patched, batch_size=batch_size,
														   shuffle=False, num_workers=0)
dataloaders_dict['notpatched'] =  torch.utils.data.DataLoader(dataset_notpatched, batch_size=batch_size,
															  shuffle=False, num_workers=0)

print("Number of clean images: {}".format(len(dataset_clean)))
print("Number of poison images: {}".format(len(dataset_poison)))


# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
# logging.info("Params to learn:")
if feature_extract:
	params_to_update = []
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			params_to_update.append(param)
			# logging.info(name)
			# print(name)
else:
	for name,param in model_ft.named_parameters():
		if param.requires_grad == True:
			# logging.info(name)
			# print(name)
			pass
# params_to_update = model_ft.parameters() # debug
# optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum = momentum)
optimizer_ft = None
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

model = model_ft.cuda(gpu)

# Train and evaluate
model, meta_dict = train_model(model, dataloaders_dict, criterion, optimizer_ft,
								  num_epochs=epochs, is_inception=(model_name=="inception"))
