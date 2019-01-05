import os
import numpy as np
from skimage import io
import scipy

def readRGBIntoNumpy(directory):
	return readImagesIntoNumpy(directory)[:,:,:,:3]

def readGrayScaleIntoNumpy(directory):
	return np.expand_dims(readImagesIntoNumpy(directory), axis = 3)

def readImagesIntoNumpy(directory):
	files = os.listdir(directory)
	images = map(lambda file: io.imread(directory + file), files)
	return np.array(images).astype(float) / 255

def saveGrayScaleImages(directory, images):
	indexes = list(range(images.shape[0]))
	map(lambda i: scipy.misc.imsave(directory + ('image%s.png' % i),
                  images[i,:,:]), indexes)
