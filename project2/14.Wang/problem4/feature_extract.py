from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd

num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(train_data, train_label), (test_data, test_label) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
	train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
	test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
	test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255
print('train_data shape:', train_data.shape)
print(train_data.shape[0], 'train samples')
print(test_data.shape[0], 'test samples')

# convert class vectors to binary class matrices
train_label = keras.utils.to_categorical(train_label, num_classes)
test_label = keras.utils.to_categorical(test_label, num_classes)

model = VGG16(weights='imagenet', include_top=False)

train_feature = []
test_feature = []
for img in train_data:
	img = np.tile(img, (9,9,3))
	img = np.expand_dims(img, axis=0)
	feature = model.predict(img)
	feature = feature.reshape(1, 7*7*512)
	train_feature.append(feature)
	
for img in test_data:
	img = np.tile(img, (9,9,3))
	img = np.expand_dims(img, axis=0)
	feature = model.predict(img)
	feature = feature.reshape(1, 7*7*512)
	test_feature.append(feature)

np.savetxt("train_feature.csv", train_feature, delimiter=",", fmt='%s')
np.savetxt("test_feature.csv", test_feature, delimiter=",", fmt='%s')

print(len(train_feature))
print(len(test_feature))