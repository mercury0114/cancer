from keras.models import load_model
from utils import readRGBIntoNumpy, saveGrayScaleImages
import numpy as np

def predict(model, xPredict):
	yPredict = model.predict(xPredict)[:,:,:,0]
	threshold = 0.5
	yPredict[yPredict < threshold] = 0
	yPredict[yPredict >= threshold] = 255
	return yPredict.astype(np.uint8)

model = load_model("./trained.h5")
testX = readRGBIntoNumpy("./testX/")
testY = predict(model, testX)
saveGrayScaleImages("./testY/", testY)
