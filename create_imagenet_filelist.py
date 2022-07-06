import configparser
import glob
import os
import sys
import random
import pdb
import tqdm

random.seed(10)
config = configparser.ConfigParser()
config.read(sys.argv[1])

options = {}
for key, value in config['dataset'].items():
	key, value = key.strip(), value.strip()
	options[key] = value

if not os.path.exists("ImageNet_data_list/poison_generation"):
	os.makedirs("ImageNet_data_list/poison_generation")
if not os.path.exists("ImageNet_data_list/finetune"):
	os.makedirs("ImageNet_data_list/finetune")
if not os.path.exists("ImageNet_data_list/test"):
	os.makedirs("ImageNet_data_list/test")

DATA_DIR = options["data_dir"]

dir_list = sorted(glob.glob(DATA_DIR + "/train/*"))

for i, dir_name in enumerate(dir_list):
	if i%50==0:
		print(i)
	filelist = sorted(glob.glob(dir_name + "/*"))
	random.shuffle(filelist)

	with open("ImageNet_data_list/poison_generation/" + os.path.basename(dir_name) + ".txt", "w") as f:
		for ctr in range(int(options["poison_generation"])):
			f.write(filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")
	with open("ImageNet_data_list/finetune/" + os.path.basename(dir_name) + ".txt", "w") as f:
		for ctr in range(int(options["poison_generation"]), len(filelist)):
			f.write(filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")

dir_list = sorted(glob.glob(DATA_DIR + "/val/*"))

for i, dir_name in enumerate(dir_list):
	if i%50==0:
		print(i)
	filelist = sorted(glob.glob(dir_name + "/*"))
	with open("ImageNet_data_list/test/" + os.path.basename(dir_name) + ".txt", "w") as f:
		for ctr in range(int(options["test"])):
			f.write(filelist[ctr].split("/")[-2] + "/" + filelist[ctr].split("/")[-1] + "\n")

