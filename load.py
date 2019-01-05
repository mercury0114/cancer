import os
import sys
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
import tensorflow as tf

acts = ['elu']

imh,imw = 256,256

def sliding_window(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

modfname = './trained.h5'

model = load_model(modfname)

sSize = 128
wSize = 256


im1 = imread('./testLarge.png')
outfname = './mask.png'
d0 = 256 - (im1.shape[0] % 256)
d1 = 256 - (im1.shape[1] % 256)

im2 = np.zeros(shape = (im1.shape[0]+d0,im1.shape[1]+d1,3),dtype = np.uint8)
im2[:im1.shape[0],:im1.shape[1],:] = im1[:, :, 0:3]
im3 = np.zeros(shape = (im2.shape[0],im2.shape[1]),dtype = np.int)

for (x,y,window) in sliding_window(im2,stepSize=sSize,windowSize=(wSize,wSize)):
    print("Dealing with x = ", x, " y = ", y)
    if window.shape[0] == wSize and window.shape[1] == wSize:
        window = window.astype(np.float) / 255.
        w = model.predict(window.reshape(1,wSize,wSize,3), batch_size=1, verbose=1)
        w[w <= 0.5] = 0
        w[w > 0.5] = 255
        w = w.astype(np.uint8)
        im3[y:y+wSize,x:x+wSize] += w.reshape(wSize,wSize)
im3 = im3[:-d0,:-d1]
im3[im3 > 255] = 255
plt.imsave(outfname,im3,cmap='gray')
plt.close()
print(''.join(["DONE with ", outfname]))

