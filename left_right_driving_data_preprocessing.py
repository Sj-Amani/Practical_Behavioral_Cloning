# Preprocess left/right side driving data and save them as a pickle file

import numpy as np
import pickle
import os

lf_ang_offset = 0.5  	# angular offset for left/right cameras
ang_offset_thr = 0.  	# angular threshold to exceed before applying offset
mov_avg_win = 1  		# moving average window

# func to implement a moving average with numpy to smooth the steering angle over time
def moving_average(a, n=3):
	if n==1:
		return a
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


print('Left/Right Side Driving Data Preprocessing ...')

for case in ('left', 'right'):
	# Get raw driving data
	with open(os.getcwd()+'/data/driving_data_%s.p' % case, mode='rb') as f:
		driving_data = pickle.load(f)

	images = driving_data['images']
	labels = driving_data['labels']

	# Calculate moving average of steering angle
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

	new_images = []
	new_labels = []
	for i, label in enumerate(labels):
		# For left/right recovery, only care about left/right camera images
		# Ignore all steering angles that are negative/positive
		if case=='left' and label[1]==1. and label[0] >= 0:
			new_label = min(1., label[0] + lf_ang_offset)
		elif case=='right' and label[1]==2. and label[0] <= 0:
			new_label = max(-1., label[0] - lf_ang_offset)
		else:
			continue

		new_images.append(images[i])
		new_labels.append(new_label)

	images = np.array(new_images)
	labels = np.array(new_labels)

	driving_data['images'] = images
	driving_data['labels'] = labels
	
	# Save the processed info as a pickle file which contains a dict: 
	# {'data': image_file_names, 'labels': adjusted_steering_angles}
	with open(os.getcwd()+'/data/preprocessed_%s.p' % case, mode='wb') as f:
		pickle.dump(driving_data, f)

print('COMPLETE!')
