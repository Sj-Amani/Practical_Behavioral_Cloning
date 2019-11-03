# Preprocess middle side driving data and save them as a pickle file

import numpy as np
import pickle
import os

mov_avg_win = 3  	# window size of moving average
lf_ang_offset = 0.15  	# angular offset for left/right cameras
ang_offset_thr = 0.  	# angular threshold to exceed before applying offset
zero_pct_val = 0.5  	# fraction of data for 0-degree steering angle
zero_angle = False  	# a flag for change data distribution to reduce frequency of 0-degree steering

# func to calculate the zero angle steering for zero_pct data
def zero_angle_steering(data_in, labels_in, zero_pct):

	total_data_size = data_in.shape[0]
	zero_idx = []  # all list indices where labels[idx]==0
	for i, label in enumerate(labels_in):
		if label[0] == 0.0:
			zero_idx.append(i)

	nonzero_data_size = total_data_size - len(zero_idx)
	zero_data_size = int(zero_pct * nonzero_data_size / (1 - zero_pct))

	# Randomly remove 0-data, such that zero_pct of new dataset is 0-data
	remove_idx = np.random.choice(zero_idx, total_data_size - zero_data_size - nonzero_data_size, replace=False)
	data = np.delete(data_in, remove_idx)
	labels = np.delete(labels_in, remove_idx, axis=0)

	return data, labels

# func to implement a moving average with numpy to smooth the steering angle over time
def moving_average(a, n=3):
	if n==1:
		return a
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n
	
# Main
print('Middle Side Driving Data Preprocessing ...')

# Get the mapping driving data
file_name= os.getcwd()+'/data/driving_data_middle.p'
with open(file_name, mode='rb') as f:
	driving_data = pickle.load(f)

images = driving_data['images']
labels = driving_data['labels']

# Find the moving average of steering angle
center_angles = labels[::3][:,0]  # 1D tensor of all steering angles from center camera
left_angles   = labels[1::3][:,0]
right_angles  = labels[2::3][:,0]

center_angles_ma = moving_average(center_angles, n=mov_avg_win)
left_angles_ma   = moving_average(left_angles, n=mov_avg_win)
right_angles_ma  = moving_average(right_angles, n=mov_avg_win)

# Throw away first few data points due to moving avg
images = images[3 * (mov_avg_win - 1):]
labels = labels[3 * (mov_avg_win - 1):]

# Replace original labels with moving average steering angles
labels[::3][:,0]  = center_angles_ma
labels[1::3][:,0] = left_angles_ma
labels[2::3][:,0] = right_angles_ma

# Calculate the zero angle steering for zero_pct data to remove bias of driving straight
if zero_angle:
	images, labels = zero_angle_steering(images, labels, zero_pct_val)

# For all left/right camera images
# If steering_angle >+/- lf_ang_offset to steering angle
new_images = []
new_labels = []
for i, label in enumerate(labels):
	if label[1] == 0.:  # center
		new_label = label[0]
	elif abs(label[0]) >= ang_offset_thr:  # only offset left/right camera angle if steering angle >= threshold
		if label[1] == 1.:  # left
			new_label = min(1., label[0] + lf_ang_offset)
		else:  # right
			new_label = max(-1., label[0] - lf_ang_offset)
	else:  # if not center camera, and steering angle < threshold, skip these data points
		continue

	new_images.append(images[i])
	new_labels.append(new_label)

images = np.array(new_images)
labels = np.array(new_labels)

driving_data['images'] = images
driving_data['labels'] = labels

# Save the processed info as a pickle file which contains a dict: {'data': image_file_names, 'labels': adjusted_steering_angles}
with open(os.getcwd()+'/data/preprocessed_middle.p', mode='wb') as f:
	pickle.dump(driving_data, f)
	
print('COMPLETE!')
