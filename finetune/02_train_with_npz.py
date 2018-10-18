# Updated KF 10/17/2018
# KF 10/17/2018

# Transfer Learning - Fine Tune
# The input data is the well-preprocessed, npz data,
# in which 'data' and corresponding 'labels' are stored.
# [Caution] The image size defined in '01_preprocessing.py'
# should be consistent with that in this script.
import numpy as np
import os
import random
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument('--directory', required=True,
				help='path to save everything')
ap.add_argument('--dataset', required=True,
                help='path to input dataset')
ap.add_argument('--pretrained', required=True,
				help='pretrained model name')
ap.add_argument('--img_size', required=True,
				help='image size option: 224 or 299')
ap.add_argument('--epochs', required=True)
ap.add_argument('--batch_size', required=True)
ap.add_argument('--learning_rate', required=True)

ap.add_argument('-m', '--model', required=True,
                help='output model file name')
ap.add_argument('-l', '--labelbin', required=True,
                help='output label binarizer file name')
ap.add_argument('-p', '--plot', type=str, default='plot.png',
                help='path to output accuracy/loss plot')
ap.add_argument('-t', '--test', action='store_true',
                help='add this flag when testing the code')
ap.add_argument('-a', '--aug', action='store_true',
                help='apply augumentation to image dataset')
args = vars(ap.parse_args())

save_dir = args['directory']
data_dir = args['dataset']
pretrained = args['pretrained']
model_pool = ('Xception', 'VGG16', 'VGG19', 'ResNet50', 'InceptionV3', 'InceptionResNetV2','MobileNet', 'DenseNet', 'NASNet', 'MobileNetV2')
if pretrained in model_pool:
	exec('from tensorflow.keras.applications import ' + pretrained)
else:
    print('Models', model_pool)    
    raise("[KF ERROR] the pre-trained model's name provided is wrong!")

# Hyper-parameters
batch_size = eval(args['batch_size'])
epochs = eval(args['epochs'])
lr = eval(args['learning_rate'])
img_size = eval(args['img_size'])
image_dim=(img_size, img_size, 3)
print("[KF INFO] Fine-tune with pre-trained model: ", pretrained)
print("[KF INFO] Training hyper-parameters:")
print("Epochs: ", epochs)
print("Batch size: ", batch_size)
print("Learning rate: ", lr)

# Load npz file
with np.load(data_dir) as npz:
    # here both data and labels should be list 
    data = npz['data']
    labels = npz['labels']
    cls_number = npz['cls_number']

# Options
if args['test']:
    data = data[:64]
    labels = labels[:64]
    batch_size = 16
aug_scale = 1
if args['aug']:
    aug_scale = 8

data = np.array(data)
labels = np.array(labels)
print('[KF INFO] data shape:')
print(data.shape)
print('[KF INFO] data matrix: {:.2f}MB'.format(data.nbytes / (1024*1000.0)))

# Binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Partition the data into training and validation dataset wiht 80% - 20% rule
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=64)

# Load the pre-trained model - with input pretrained model's name
if pretrained == 'VGG16':
    if img_size != 224:
        raise("[KF ERROR] For %s model, the input image size is not 224!" % pretrained)
    conv = VGG16(weights='imagenet', include_top=False, input_shape=image_dim)

elif pretrained == 'MobileNetV2':
    if img_size != 224:
        raise("[KF ERROR] For %s model, the input image size is not 224!" % pretrained)
    conv = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_dim)

elif pretrained == 'InceptionResNetV2':
    if img_size != 299:
        raise("[KF ERROR] For %s model, the input image size is not 299!" % pretrained)
    conv = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=image_dim)
else:
    raise("[KF INFO] Cannot load the pre-trained model, add code snippet ...")

print("[KF INFO] The pretrained model %s's convolutional part is loaded ..." % pretrained)

# Freeze the required layers from training
for layer in conv.layers:
    layer.trainable = False
    print(layer, layer.trainable)

# initialize the model
model = models.Sequential()

# Add the vgg convolutional base model first
model.add(conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(cls_number, activation='softmax'))

# Show model summary
model.summary()

model.compile(optimizer=optimizers.Adam(lr=lr, decay=lr/epochs),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Construct image generator f or data augmentation
datagen = ImageDataGenerator(rescale=1./255)
if args['aug']:
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=25,
                    width_shift_range=0.1, height_shift_range=0.1,
                    shear_range=0.2, zoom_range=0.2,
                    horizontal_flip=True, fill_mode='nearest')

# Happy training ...
H = model.fit_generator(
        datagen.flow(trainX, trainY, batch_size=batch_size),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // batch_size * aug_scale,
        epochs=epochs)

# Create save directory if it's not there
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# save the model to disk
model.save(os.path.join(save_dir, args['model']))
print('[KF INFO] model saved!')
# save the label binarizer to disk
with open(os.path.join(save_dir, args['labelbin']), 'wb') as f:
    f.write(pickle.dumps(lb))
    print('[KF INFO] label binarizer saved!')

# Check Performance - plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(os.path.join(save_dir, args["plot"]))

print('[KF INFO] Training completed! Take a break here ...')

