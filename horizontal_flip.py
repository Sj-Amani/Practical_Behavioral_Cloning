# Flip the images horizontally and save the driving_log_flipped.csv

import numpy as np
from PIL import Image

# Read original csv
with open('driving_log.csv', mode='r') as f:
	lines = f.readlines()

# Flip & save images, create new csv
new_csv = ''
for line in lines:
	fields = line.split(',')

	image_paths = (fields[0], fields[1][0:], fields[2][0:])  # strip leading space for L/R image paths
	new_image_paths = []

	# For each center/left/right camera image, flip it horizontally
	for image_path in image_paths:
		# Read image and flip
		image = Image.open(image_path)
		image_f = image.transpose(Image.FLIP_LEFT_RIGHT)

		# Save new image
		new_image_path = image_path.split('.')[0] + '_flipped.jpg'
		image_f.save(new_image_path)

		new_image_paths.append(new_image_path)

	new_steering_angle = -1. * float(fields[3])

	new_line = '%s, %s, %s, %f,%s,%s,%s' % (*new_image_paths, new_steering_angle, *fields[-3:])
	new_csv += new_line

with open('driving_log_flipped.csv', mode='w') as f:
	f.write(new_csv)
