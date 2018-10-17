
# KF 09/19/2018

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
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
ap.add_argument("-t", "--test", action="store_true",
        help="add this flag when testing")
ap.add_argument("-a", "--aug", action="store_true",
        help="add this flag when applying augmentation")
args = vars(ap.parse_args())

data_dir = args['dataset']
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-4
BS = 32
DIM_SIZE = 224
IMAGE_DIMS = (DIM_SIZE, DIM_SIZE, 3)

# initialize the data and labels
data = []
labels = []

# Load npz file
with np.load(data_dir) as npz:
    # here both data and labels should be list 
	data = npz['data']
	labels = npz['labels']
cls_number = 4 

# Options
if args['test']:
    data = data[:64]
    labels = labels[:64]
    batch_size = 4 
aug_scale = 1 
if args['aug']:
    aug_scale = 4 

data = np.array(data)
labels = np.array(labels)
print("[KF INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
datagen = ImageDataGenerator(rescale=1./255)
if args['aug']:
    datagen = ImageDataGenerator(rescale=1./255, 
                rotation_range=25, width_shift_range=0.1,
                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[KF INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# Show model summary
model.summary()

model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[KF INFO] training network...")
H = model.fit_generator(
	datagen.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[KF INFO] serializing network...")
model.save(args["model"])
print("[KF INFO] model saved!")

# save the label binarizer to disk
print("[KF INFO] serializing label binarizer...")
with open(args["labelbin"], "wb") as f:
    f.write(pickle.dumps(lb))
    print("[KF INFO] label binarizer saved!")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
