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
import datetime
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
ap.add_argument('--dataset', required=True,
				help='path to the training dataset directory')
ap.add_argument('--save_dir', required=True,
				help='path to save the trained model')
ap.add_argument('--training_mode', type=str, default='full',
                help="select mode in 'full' and 'trial', if trial mode, all parameters are set to small scales")
ap.add_argument('--pretrained', required=True,
				help='pretrained model name')
ap.add_argument('--img_size', required=True,
				help='image size option: 224 or 299')
ap.add_argument('--epochs', required=True)

ap.add_argument('--batch_size', required=True)

ap.add_argument('--optimizer', required=True)

ap.add_argument('--model_file_name', required=True,
                help='output model file name')
ap.add_argument('--plot', type=str, default='plot.png',
                help='path to output accuracy/loss plot')
ap.add_argument('--trainset_option', required=True,
                help="'train' for train set only, 'concat' for concatenated dataset")
ap.add_argument('--freeze', action='store_true',
                help="add this parameter if you want to freeze the convolutional layer's parameters from training.")

args = vars(ap.parse_args())

# Path parameters
data_dir = args['dataset']
save_dir = args['save_dir']
model_file_name = args['model_file_name']

# Training mode string
training_mode = args['training_mode']

# Basic training parameters
batch_size = eval(args['batch_size'])
epochs = eval(args['epochs'])
img_size = eval(args['img_size'])
img_dim=(img_size, img_size, 3)

# Class numbers and image numbers 
trainset_option = args['trainset_option'] # Either 'train' or 'concat'
cls_number = 61
train_img_num = 31718
val_img_num = 4540
if trainset_option == 'concat':
    train_img_num = train_img_num + val_img_num

# Pretrained model name selection
pretrained = args['pretrained']
model_pool = ('Xception', 'VGG16', 'VGG19', 'ResNet50', 'InceptionV3', 'InceptionResNetV2','MobileNet', 'DenseNet', 'NASNet', 'MobileNetV2')
if pretrained in model_pool:
	exec('from tensorflow.keras.applications import ' + pretrained)
else:
    print('Models', model_pool)    
    raise("[KF ERROR] the pre-trained model's name provided is wrong!")

# FC parameters
fc_size = 1024
dropout_rate = 0.5

# Initialize the optimizer for the model
optimizer_name = args['optimizer']
decay = 0.0
if optimizer_name == 'rmsprop':
    init_lr = 0.045
    #decay = 0.9
    decay = 0.08
    optimizer = optimizers.RMSprop(lr=init_lr, epsilon=1.0, decay=decay)

elif optimizer_name == 'adam':
    init_lr = 0.0001
    #decay = init_lr/epochs
    decay = 0.003
    optimizer = optimizers.Adam(lr=init_lr, decay=decay)
 
elif optimizer_name =='nadam':
    init_lr = 0.002
    decay = 0.06
    optimizer = optimizers.Nadam(lr=init_lr, schedule_decay=decay)


# Trial mode parameter config
if training_mode == 'trial': 
    batch_size = 4
    #img_size = 150
    #img_dim = (img_size, img_size, 3)
    train_img_num = train_img_num // 100
    val_img_num = val_img_num // 20

# Added - KF 11/08/2018
#concat_img_num = train_img_num + val_img_num

print('')
print('=================================================================')
print('AI Challenger Competition Crop Disease Recognition Model Training')
print('=================================================================')
print('')
print('Scripts written by:')
print('  Zhu Kefeng')
print('  zkf1985@gmail.com')
print('')
print('Training on date:')
print('  ' + datetime.datetime.today().strftime('%Y-%m-%d'))
print('')
print('=================================================================')
print('                   1. TRAINING OVERVIEW') 
print('=================================================================')
print('')
print('< TRAINING MODE >')
print('-----------------------------------------------------------------')
print('Training mode        :', training_mode) 
print('-----------------------------------------------------------------')
print('')
print('< DATA INFO >')
print('-----------------------------------------------------------------')
print("Class number         :", cls_number)
print("Trainset option      :", trainset_option)
print("Train data size      :", train_img_num)
print("Val data size        :", val_img_num)
print('-----------------------------------------------------------------')
print('')
print('< MODEL INFO >')
print('-----------------------------------------------------------------')
print("Pre-trained model    :", pretrained)
print("Freeze conv layers   :", args['freeze'])
print("Image size           :", img_size)
print("Epochs               :", epochs)
print("Batch size           :", batch_size)
print('-----------------------------------------------------------------')
print("FC layer size        :", fc_size)
print("Dropout rate         :", dropout_rate)
print('-----------------------------------------------------------------')
print("Optimizer            :", optimizer_name)
print("Initial LRate        :", init_lr)
print("LR decay rate        :", decay)
print('-----------------------------------------------------------------')

print('')
print('=================================================================')
print('                   2. BUILD THE MODEL') 
print('=================================================================')

print('[KF INFO] Loading pre-trained model ...')
if pretrained == 'VGG16':
    if img_size != 224:
        raise("[KF ERROR] For %s model, the input image size is not 224!" % pretrained)
    conv = VGG16(weights='imagenet', include_top=False, input_shape=img_dim)
elif pretrained == 'VGG19':
    if img_size != 224:
        raise("[KF ERROR] For %s model, the input image size is not 224!" % pretrained)
    conv = VGG19(weights='imagenet', include_top=False, input_shape=img_dim)
elif pretrained == 'MobileNetV2':
    if img_size != 224:
        raise("[KF ERROR] For %s model, the input image size is not 224!" % pretrained)
    conv = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_dim)

elif pretrained == 'InceptionResNetV2':
    #if img_size != 299:
    #    raise("[KF ERROR] For %s model, the input image size is not 299!" % pretrained)
    conv = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=img_dim)
elif pretrained == 'InceptionV3':
    if img_size != 299:
        raise("[KF ERROR] For %s model, the input image size is not 299!" % pretrained)
    conv = InceptionV3(weights='imagenet', include_top=False, input_shape=img_dim)
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
model.add(layers.Dense(fc_size, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(cls_number, activation='softmax'))
print('[KF INFO] The FC layers are added to the model.')

# Show model summary
model.summary()

# Compile the model
model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
print('[KF INFO] The model is compiled successfully!')

print('')
print('=================================================================')
print('                   3. TRAIN THE MODEL') 
print('=================================================================')

# Construct image generator with augmentation
# Update - KF 11/08/2018
# Only with concatenated dataset, without any valication dataset print('')
#print('[KF INFO] Create train/validation data generator ...')
print('[KF INFO] Create train data generator ...')
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=25,
				width_shift_range=0.1, height_shift_range=0.1,
				shear_range=0.2, zoom_range=0.2,
				horizontal_flip=True, fill_mode='nearest')
#train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# trainset folder name depends on the parameter trainset_option
print("[KF INFO] Caution: Training dataset:", trainset_option)
train_generator = train_datagen.flow_from_directory(
		os.path.join(data_dir, trainset_option),
		target_size=(img_size, img_size),
		batch_size=batch_size,
		class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
		os.path.join(data_dir, 'val'),
		target_size=(img_size, img_size),
		batch_size=batch_size,
		class_mode='categorical')

# Create save directory if it's not there
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Save the label-indeces dictionary to an npz file
np.savez(os.path.join(save_dir, 'labels'), class_idx=train_generator.class_indices, class_label=train_generator.classes)

print('')
print('[KF INFO] Start training ...')

# KF 11/07/2018
# Add callbacks
# Check point
checkpointer = keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, 'ckpt-weight-best.hdf5'), 
                                save_best_only=True)
# Tensorboard
#`tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join(save_dir, 'logs'))

# Reduce learning rate on plateau
#reducelronplateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

callbacks_list = [checkpointer]

H = model.fit_generator(
		train_generator,
		steps_per_epoch=train_img_num // batch_size,
		epochs=epochs,
                verbose=1,
                callbacks=callbacks_list,
                validation_data=val_generator,
                validation_steps=val_img_num // batch_size)

print('[KF INFO] Training completed!!!')
print('-----------------------------------------------------------------')

# save the model to disk
model.save(os.path.join(save_dir, model_file_name))
print('[KF INFO] Model saved!')

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
print('[KF INFO] Learning plot saved!')

#plt.figure()
#plt.plot(np.arange(0, N), H.history["lr"], label="learning_rate")
#plt.title("Training Loss and Accuracy")
#plt.xlabel("Epoch #")
#plt.ylabel("Learning Rate")
#plt.legend(loc="upper left")
#plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
#print('[KF INFO] Learning rate plot saved!')





