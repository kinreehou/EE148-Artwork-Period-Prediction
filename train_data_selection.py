import os
import pandas as pd
import glob 
import numpy as np
from PIL import Image

bad_files = ['72255.jpg', '92899.jpg', '50420.jpg', '95347.jpg', '81823.jpg', '33557.jpg', 
             '98873.jpg','79499.jpg','95010.jpg','101947.jpg','41945.jpg']
df = pd.read_csv('train_info.csv')
df = df.dropna()
WSI_MASK_PATH = 'train_1_sample_30/'
paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.jpg'))
#print(paths) 

train_data = []
for path in paths:
	img_name = path.split('/')[1]
	if img_name in bad_files:
		print(img_name)
		continue
	#print(img_name)
	date = df.loc[df['filename']==img_name]['date'].tolist()
	if date:
		d = date[0]
		if (d[0] == 'c'):
			d = d[2:]
		while not d[0].isdigit():
			d = d[1:]
			#print(d)
		train_data.append([img_name, int(float(d))])
 
train1_df = pd.DataFrame(np.array(train_data), columns=['filename','date'])
#train1_df.head
train1_df.to_csv('train_1_info_30_percent.csv', index = False, header=True)