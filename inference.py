# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend
import numpy as np
import pandas as pd
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
# ap.add_argument("-d", "--data", required=True,
# 	help="path to input image")
args = vars(ap.parse_args())

def f1_score(y_true, y_pred, beta=2):
	# clip predictions
	y_pred = backend.clip(y_pred, 0, 1)
	# calculate elements
	tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
	fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
	fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
	# calculate precision
	p = tp / (tp + fp + backend.epsilon())
	# calculate recall
	r = tp / (tp + fn + backend.epsilon())
	# calculate fbeta, averaged across each class
	bb = beta ** 2
	f1_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
	return f1_score

print('[INFO] Loading Model..')
mlb = pickle.loads(open(args['labelbin'], "rb").read())
model = load_model(args['model'], custom_objects={'f1_score': f1_score})

print('[INFO] Model loaded')


print('[INFO] Loading Dataset..')
folder_path = args['dir']
amazing_labels = []
images = []
for img in os.listdir(folder_path)[:10]:
    image = os.path.join(folder_path, img)
    image = cv2.imread(image)
#     output = imutils.resize(image, width=400)
    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)
    amazing_labels.append(mlb.inverse_transform(proba.round()))
    images.append(img)
    
for img in os.listdir(folder_path):
    image = os.path.join(folder_path, img)
    image = cv2.imread(image)

    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)
    amazing_labels.append(mlb.inverse_transform(proba.round()))
    images.append(img)
    
print('[INFO] Dataset loaded')


print(amazing_labels[3])
print(amazing_labels[4])
#creating a dataframe of the outputs.
data = {'image' : images, 'label' : amazing_labels}
anno = pd.DataFrame(data)
for i in range(anno.shape[0]):
    anno.image[i] = anno.image[i][:-4]
    
print('[INFO] Creating csv file...')
anno_1 = pd.read_csv('annotation.csv')
new_anno = anno.join(anno_1.set_index('image'), on='image')
new_anno = new_anno[['image', 'label']]

new_anno.to_csv('annotations_new.csv', index = False)

print('[INFO] csv file created.')
print('Check the same directory as command line.')