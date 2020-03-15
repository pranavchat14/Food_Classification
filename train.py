# USAGE
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from utilities.smallervggnet import SmallerVGGNet
from sklearn.metrics import f1_score
from keras import backend
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import random
import pickle
import cv2
import ast
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 200
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# removing string commas to facilitate sklearn's MultiBinarizer
my_label = pd.read_csv('/gdrive/My Drive/Orbuculum/annotation.csv')
my_label['new_labels'] = my_label.labels.apply(lambda s: list(ast.literal_eval(s)))
new_labels = []

# removing repeated instances of labels
for i in range(1400):
    p =  list(set(my_label.new_labels[i]))
    new_labels.append(p)
my_label['new_labels'] = new_labels

# initialize the data and labels
data = []
labels = []

# loop over the input images
for i in my_label.index:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread('data/{}.jpg'.format(my_label['image'][i]))
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
	l = label = my_label['new_labels'][i]
	labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.3, random_state=42)

# partition the data into training and testing splits using 50% of
# the data for  and the validation remaining 50% for testing
(validX, testX, validY, testY) = train_test_split(testX,
	testY, test_size=0.5, random_state=42)

# construct the image generator for data augmentation
train_datagen = ImageDataGenerator(rotation_range=90, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

test_datagen = ImageDataGenerator()

train_datagen.mean = [123.68, 116.779, 103.939]
test_datagen.mean = [123.68, 116.779, 103.939]

train_it = train_datagen.flow(trainX, trainY, batch_size=128)
valid_it = test_datagen.flow(validX, validY, batch_size=128)
test_it = test_datagen.flow(testX, testY, batch_size=128)


#initialize the model using a sigmoid activation as the final layer
#in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

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

# compile the model using binary cross-entropy
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=[f1_score])


csv_logger = CSVLogger('logs.txt')


# train the network
print("[INFO] training network...")

mcp_save = ModelCheckpoint(args["model"], save_best_only=True, verbose=1, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, workers=1, use_multiprocessing=True, verbose=1, min_delta=1e-4, mode='min')

H = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                        validation_data=valid_it, validation_steps=len(valid_it), epochs=EPOCHS, verbose=1, 
                        callbacks= [csv_logger, mcp_save, reduce_lr_loss])


print("[INFO] evaluating model..")

model.evaluate(test_it, steps=len(test_it), batch_size=None, verbose=1, max_queue_size=10)

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

plt.plot(H.history['f1_score'])
plt.plot(H.history['val_f1_score'])
plt.title('Model f1_score')
plt.ylabel('f1_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('f1_score_plot.png')

# Plot training & validation loss values
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss_plot.png')
