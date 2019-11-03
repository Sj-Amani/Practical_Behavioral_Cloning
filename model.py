# Read the preprocessed data and make a trained neural network model and save it

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils import plot_model
#from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split
import time
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
tf.python_io.control_flow_ops = tf


H, W, CH = 160, 320, 3 	# height, width, channels
epoch_n = 10		# epoch number
batch_sz = 16		# batch size
learning_r = 1e-4	# lerning rate	
L2_regulize_scale = 0.	# L2 regulizer scale allow to apply penalties on layer parameters or layer activity during optimization 

# func to generate batch
def generator(data, labels, batch_size):
	
	# Input:
	# 	data: List of strings containing the path of image files
	#	labels: List of steering angles
	#	batch_size: Size of the batch to generate
	#
	# Output:
	#	A tuple (X_batch, y_batch), where:
	#		X_batch: Batch of images, a tensor of shape (batch_size, H, W, CH)
	#		y_batch: Batch of steering angles
	
	start = 0
	end = start + batch_size
	n = data.shape[0]

	while True:
		# Read image data into memory as-needed
		image_files  = data[start:end]
		images = []
		for image_file in image_files:
			# Resize image, create numpy array representation
			image_file = '/'+ image_file
			image = Image.open(image_file).convert('RGB')
			image = image.resize((W, H), Image.ANTIALIAS)
			image = np.asarray(image, dtype='float32')
			images.append(image)
		images = np.array(images, dtype='float32')

		X_batch = images
		y_batch = labels[start:end]
		start += batch_size
		end += batch_size
		if start >= n:
			start = 0
			end = batch_size

		yield (X_batch, y_batch)

# func to get the neural network model
def get_model():

	ch, row, col = CH, H, W  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
		input_shape=(row, col, ch),
		output_shape=(row, col, ch)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', W_regularizer=l2(L2_regulize_scale)))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_regulize_scale)))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_regulize_scale)))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512, W_regularizer=l2(0.)))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1, W_regularizer=l2(0.)))

	model.compile(optimizer=Adam(lr=learning_r), loss='mean_squared_error')

	return model

# func to train the neural network model
def nn_model_training():
	# Load driving data
	with open('data/final_preprocessed_driving_data.p', mode='rb') as f:
		driving_data = pickle.load(f)

	data, labels = driving_data['images'], driving_data['labels']
	X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=0)

	# Get model
	model = get_model()
	model.summary()

	# Visualize model and save it to disk
	#plot_model(model, to_file='results/model.png', show_shapes=True, show_layer_names=False)
	#plot(model, to_file='results/model.png', show_shapes=True, show_layer_names=False)
	#print('Saved model visualization at results/model.png')

	# Instantiate generators
	train_gen = generator(X_train, y_train, batch_sz)
	val_gen = generator(X_val, y_val, batch_sz)

	train_start_time = time.time()

	# Train model
	#h = model.fit_generator(generator=train_gen, samples_per_epoch=X_train.shape[0], epochs=epoch_n, validation_data=val_gen, validation_steps=X_val.shape[0])
	h = model.fit_generator(validation_data=val_gen, validation_steps=X_val.shape[0], generator=train_gen, epochs=epoch_n, steps_per_epoch=X_train.shape[0])
	history = h.history

	total_time = time.time() - train_start_time
	print('Total training time: %.2f sec (%.2f min)' % (total_time, total_time/60))

	# Save model architecture to model.json, model weights to model.h5
	json_string = model.to_json()
	with open('model.json', 'w') as f:
		f.write(json_string)
	model.save_weights('model.h5')

	# Save training history
	with open('train_hist.p', 'wb') as f:
		pickle.dump(history, f)
	
	print('History saved as train_hist.p')
	print('Model architecture saved as model.json')
	print('Model weights saved as model.h5')

if __name__ == '__main__':
	print('Start training the model ...')
	
	nn_model_training()
	
	print('COMPLETE!')

