import os
import shutil
import glob
import random

ppt = 0.3
file_list = os.listdir('train_1/') # dir is your directory path
number_files = len(file_list)
l = int(number_files*ppt)
to_be_moved = random.sample(glob.glob("train_1/*.jpg"), l)

for f in enumerate(to_be_moved, 1):
	name = f[1].split('/')[-1]
	print(name)
	dest = os.path.join("train_1_sample_30/")
	if not os.path.exists(dest):
		os.makedirs(dest)
	shutil.copy(f[1], dest)