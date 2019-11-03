# Merge the left, middle, and right side driving data and save them as a pickle file

import numpy as np
import pickle
import os

print('Merging the all preprocessed data [left + middle + right] ...')

with open(os.getcwd()+'/data/preprocessed_middle.p', 'rb') as f:
	middle = pickle.load(f)

with open(os.getcwd()+'/data/preprocessed_left.p', 'rb') as f:
	left = pickle.load(f)

with open(os.getcwd()+'/data/preprocessed_right.p', 'rb') as f:
	right = pickle.load(f)

images = np.concatenate((middle['images'], left['images'], right['images']))
labels = np.concatenate((middle['labels'], left['labels'], right['labels']))

driving_data = {'images': images, 'labels': labels}
with open(os.getcwd()+'/data/final_preprocessed_driving_data.p', mode='wb') as f:
	pickle.dump(driving_data, f)

print('COMPLETE!')
