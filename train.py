import h5py
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import readRGBIntoNumpy, readGrayScaleIntoNumpy
from keras.layers.merge import concatenate
from keras.layers import Conv2DTranspose, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout
from keras.layers import Input
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import numpy as np
import random
random.seed(123)

def addConvolutionalLayer(layer, newDepth = None):
	if newDepth is None:
		newDepth = layer.get_shape().as_list()[3] * 2
	newLayer = Conv2D(newDepth, kernel_size = (3, 3),
					activation = 'relu', padding='same') (layer)
	newLayer = Dropout(0.2) (newLayer)
	return Conv2D(newDepth, kernel_size = (3, 3),
			activation = 'relu', padding='same') (newLayer)

def addReverseLayer(layer, parallel):
	upLayer = UpSampling2D(2)(layer)
	depth = upLayer.get_shape().as_list()[3] / 2
	newLayer = Conv2DTranspose(depth, kernel_size = (3, 3),
					activation = 'relu', padding='same') (upLayer)
	newLayer = concatenate([parallel, newLayer])
	newLayer = Conv2DTranspose(depth, kernel_size = (3, 3),
					activation = 'relu', padding='same') (newLayer)
	return Conv2DTranspose(depth, kernel_size = (3, 3),
					activation = 'relu', padding='same') (newLayer)

def buildModel():
	s = Input((256, 256, 3))
	d1 = addConvolutionalLayer(s, 32)
	d2 = addConvolutionalLayer(MaxPooling2D(2)(d1))
	d3 = addConvolutionalLayer(MaxPooling2D(2)(d2))
	d4 = addConvolutionalLayer(MaxPooling2D(2)(d3))
	d5 = addConvolutionalLayer(MaxPooling2D(2)(d4))
	u4 = addReverseLayer(d5, d4)
	u3 = addReverseLayer(u4, d3)
	u2 = addReverseLayer(u3, d2)
	u1 = addReverseLayer(u2, d1)
	o = Conv2DTranspose(1, kernel_size = (3, 3), activation='relu', padding='same')(u1)
	model = Model(inputs = [s], outputs = [o])
	model.compile(optimizer='adam', loss='binary_crossentropy',
				  metrics=['accuracy'])
	return model

model = buildModel()
print(model.summary())

trainX = readRGBIntoNumpy("./trainX/")
trainY = readGrayScaleIntoNumpy("./trainY/")

earlyStopper = EarlyStopping(monitor='val_loss', patience=35, verbose=1)
checkPointer = ModelCheckpoint('trained.h5', verbose=1, save_best_only=True)
model.fit(trainX, trainY, validation_split = 0.2, batch_size = 8,
		  epochs = 500, callbacks=[earlyStopper, checkPointer])
model.save('trained.h5')
