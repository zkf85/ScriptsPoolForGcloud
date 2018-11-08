# 05_train_concat.py

# Update - KF 11/08/2018 
# Merge the training set and the validation set
# Train it without any validation for the final submission

# KF 11/05/2018

# AI Challenger competition

# Main training script for training
import numpy as np
import math
import os
import json
import argparse
import pickle
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 

#from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from sklearn.preprocessing import LabelBinarizer

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
				help='path to the training dataset directory')
ap.add_argument('-s', '--save_dir', required=True,
				help='path to save the trained model')
ap.add_argument('--pretrained', required=True,
				help='pretrained model name')
ap.add_argument('--img_size', required=True,
				help='image size option: 224 or 299')
ap.add_argument('--epochs', required=True)
ap.add_argument('--batch_size', required=True)
ap.add_argument('--learning_rate', required=True)

ap.add_argument('-m', '--model', required=True,
                help='output model file name')
#ap.add_argument('-l', '--labelbin', required=True,
#                help='output label binarizer file name')
ap.add_argument('-p', '--plot', type=str, default='plot.png',
                help='path to output accuracy/loss plot')
ap.add_argument('-t', '--test', action='store_true',
                help='add this flag when testing the code')
ap.add_argument('--freeze', action='store_true')
args = vars(ap.parse_args())

data_dir = args['dataset']
save_dir = args['save_dir']
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
cls_number = 61
train_img_num = 31718
val_img_num = 4540

# Test option
if args['test']:
	batch_size = 16
	train_img_num = train_img_num // 100
	val_img_num = val_img_num // 100

# Added - KF 11/08/2018
concat_img_num = train_img_num + val_img_num

print('')
print('============================================================')
print('                   1. HYPERPARAMETERS') 
print('============================================================')
print("[KF INFO] Training hyper-parameters:")
print('')
print("Pre-trained model :", pretrained)
print("Image size        :", img_size)
print("Epochs            :", epochs)
print("Batch size        :", batch_size)
print("Learning rate     :", lr)
print("Class number      :", cls_number)
#print("Train data size   :", train_img_num)
#print("Val data size     :", val_img_num)
print("Train data size   :", concat_img_num)


print('')
print('============================================================')
print('                   2. BUILD MODEL') 
print('============================================================')

print('[KF INFO] Loading pre-trained model ...')
if pretrained == 'VGG16':
    if img_size != 224:
        raise("[KF ERROR] For %s model, the input image size is not 224!" % pretrained)
    conv = VGG16(weights='imagenet', include_top=False, input_shape=image_dim)
elif pretrained == 'VGG19':
    if img_size != 224:
        raise("[KF ERROR] For %s model, the input image size is not 224!" % pretrained)
    conv = VGG19(weights='imagenet', include_top=False, input_shape=image_dim)
elif pretrained == 'MobileNetV2':
    if img_size != 224:
        raise("[KF ERROR] For %s model, the input image size is not 224!" % pretrained)
    conv = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_dim)

elif pretrained == 'InceptionResNetV2':
    #if img_size != 299:
    #    raise("[KF ERROR] For %s model, the input image size is not 299!" % pretrained)
    conv = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=image_dim)
elif pretrained == 'InceptionV3':
    if img_size != 299:
        raise("[KF ERROR] For %s model, the input image size is not 299!" % pretrained)
    conv = InceptionV3(weights='imagenet', include_top=False, input_shape=image_dim)
else:
    raise("[KF INFO] Cannot load the pre-trained model, add code snippet ...")

print("[KF INFO] The pretrained model %s's convolutional part is loaded ..." % pretrained)

# Freeze the required layers from training
for layer in conv.layers:
    if args['freeze']:
        layer.trainable = False
        print(layer, layer.trainable)

# initialize the model
model = models.Sequential()

# Add the vgg convolutional base model first
model.add(conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(cls_number, activation='softmax'))
print('[KF INFO] The DNN part of the model is added.')

# Show model summary
model.summary()

model.compile(optimizer=optimizers.Adam(lr=lr, decay=lr/epochs),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print('')
print('============================================================')
print('                   3. TRAIN THE MODEL') 
print('============================================================')

# Construct image generator with augmentation
# Update - KF 11/08/2018
# Only with concatenated dataset, without any valication dataset
print('')
#print('[KF INFO] Create train/validation data generator ...')
print('[KF INFO] Create train data generator ...')
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=25,
				width_shift_range=0.1, height_shift_range=0.1,
				shear_range=0.2, zoom_range=0.2,
				horizontal_flip=True, fill_mode='nearest')
#train_datagen = ImageDataGenerator(rescale=1./255)
#val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
		os.path.join(data_dir, 'concat'),
		target_size=(img_size, img_size),
		batch_size=batch_size,
		class_mode='categorical')

#val_generator = val_datagen.flow_from_directory(
#		os.path.join(data_dir, 'val'),
#		target_size=(img_size, img_size),
#		batch_size=batch_size,
#		class_mode='categorical')
np.savez(os.path.join(save_dir, 'labels'), class_idx=train_generator.class_indices, class_label=train_generator.classes)

print('')
print('[KF INFO] Start training ...')

# KF 11/07/2018
# Check point
checkpointer = keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, 'ckpt-weight-best.hdf5'), 
                                save_best_only=True)

callbacks_list = [checkpointer]

H = model.fit_generator(
		train_generator,
		steps_per_epoch=concat_img_num // batch_size,
		epochs=epochs)

print('[KF INFO] Training completed!!!')
print('------------------------------------------------------------')

# Create save directory if it's not there
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# save the model to disk
model.save(os.path.join(save_dir, args['model']))
print('[KF INFO] Model saved!')
# save the label binarizer to disk
#with open(os.path.join(save_dir, args['labelbin']), 'wb') as f:
#    f.write(pickle.dumps(lb))
#    print('[KF INFO] label binarizer saved!')

# Check Performance - plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(os.path.join(save_dir, args["plot"]))

plt.figure()
plt.plot(np.arange(0, N), H.history["lr"], label="learning_rate")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Learning Rate")
plt.legend(loc="upper left")
plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
print('[KF INFO] Learning rate plot saved!')





