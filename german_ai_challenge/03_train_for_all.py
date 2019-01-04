# 02_smallvgg_balanced_generator.py
#################################################################
# Alibaba Cloud German AI Challenge 2018
# KF 2018/12/06
#    2018/12/07
#    2018/12/10 - add generators
#    2018/12/17 - make a balanced generator
#    2019/01/03 - add channel option "s1_ch56"
#    2019/01/04   "s1_ch56" doesn't perform good, change it to "s1_ch78"
#                 "s1_ch78" not working well, change it to "s1_ch5678"
#
#################################################################
import os
import random
import numpy as np
import h5py
import tensorflow as tf
from datetime import datetime
import csv

# Print title with double lines with text aligned to center
def print_title(title):
    print('')
    print('=' * 65)
    print(' ' * ((65 - len(title))//2 - 1), title)
    print('=' * 65)

#################################################################
# I. Parameters
#################################################################
# Set Paths
base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')
train_filename = 'training.h5'
val_filename = 'validation.h5'
round1_test_filename = 'round1_test_a_20181109.h5'

# Set Train mode: 'real' or 'test'
train_mode = 'real'
#train_mode = 'test'

# Set real epoch for the training process
#epochs = 3
#epochs = 100 
epochs = 200

# Set batch_size 
#batch_size = 256
batch_size = 128
#batch_size = 64
#batch_size = 32
#batch_size = 16
#batch_size = 8
#batch_size = 4

# Initial learning rate 
#lr = 0.001
lr = 0.0003
#lr = 0.0001

# Early Stopping patience:
#early_stopping_patience = 10
#early_stopping_patience = 20
early_stopping_patience = 30

# ReduceLRPlateau patience:
#reduce_lr_patience = 6
#reduce_lr_patience = 8
reduce_lr_patience = 10

# Set data channel: 'full' or 's2_rgb'
#data_channel = 'full'
#data_channel = 's2_rgb'
#data_channel = 's1'
data_channel = 's2'
#data_channel = 's1_ch5678'

# Set data generating mode: 'original' or 'balanced'
# if original, class_weight should be set
data_gen_mode = 'original'
#data_gen_mode = 'balanced'
#data_gen_mode = 'val_dataset_only'

# Set model name
#model_name = 'KFSmallerVGGNet'
#model_name = 'KFDummy'
#model_name = 'KFResNet18'
#model_name = 'KFResNet34'
model_name = 'KFResNet50'
#model_name = 'KFResNet101'
#model_name = 'KFResNet152'

#################################################################
# II. Load Data and Generator
#################################################################
from kfdata.KFGermanData import GermanData

# Parameter dictionary for initializing GermanData instance
param_dict = {}
# Add parameters to param_dict
param_dict['base_dir'] = base_dir
param_dict['train_filename'] = train_filename
param_dict['val_filename'] = val_filename
param_dict['round1_test_filename'] = round1_test_filename
param_dict['train_mode'] = train_mode
param_dict['batch_size'] = batch_size
param_dict['data_channel'] = data_channel
param_dict['data_gen_mode'] = data_gen_mode

# Create GermanData class instance
german_data = GermanData(param_dict)

# Get train/val generators
train_gen = german_data.train_gen
val_gen = german_data.val_gen

# Set model saving Path 
#   requires: model_name, cur_date, epochs, train_size
cur_date = datetime.now()
res_root_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/results')
res_folder_name = 'model-%d%02d%02d-%s-epochs-%d-trainsize-%d-channels-%s' % (cur_date.year, cur_date.month, cur_date.day, model_name, epochs, german_data.train_size, data_channel)
if not os.path.exists(os.path.join(res_root_dir, res_folder_name)):
    os.makedirs(os.path.join(res_root_dir, res_folder_name))

#----------------------------------------------------------------
# Print Training Parameters
#----------------------------------------------------------------
print('')
print_title("Training Parameters")
print("Train Mode       :", german_data.train_mode)
print("Data Channel     :", german_data.data_channel)
print("Data Gen Mode    :", german_data.data_gen_mode)
print("Train Size       :", german_data.train_size)
print("Validation Size  :", german_data.val_size)
print("Batch Size       :", german_data.batch_size)
print("Data Dimension   :", german_data.data_dimension)
print("Epochs           :", epochs)
print("Initial LR       :", lr)
print('-'*65)
print("Model Name       :", model_name)
print("Model Saving Directory:")
print(os.path.join(res_root_dir, res_folder_name))
print('-'*65)


#################################################################
# III. Build the Model
#################################################################
from kfmodels.kfmodels import KFSmallerVGGNet, KFDummy,KFResNet18, KFResNet34, KFResNet50, KFResNet101, KFResNet152
from tensorflow.keras import optimizers

# Select model with model name
if model_name == 'KFSmallerVGGNet':
    model = KFSmallerVGGNet.build(german_data.data_dimension)
elif model_name == 'KFDummy':
    model = KFDummy.build(german_data.data_dimension)
elif model_name == 'KFResNet18':
    model = KFResNet18.build(german_data.data_dimension)
elif model_name == 'KFResNet34':
    model = KFResNet34.build(german_data.data_dimension)
elif model_name == 'KFResNet50':
    model = KFResNet50.build(german_data.data_dimension)
elif model_name == 'KFResNet101':
    model = KFResNet101.build(german_data.data_dimension)
elif model_name == 'KFResNet152':
    model = KFResNet152.build(german_data.data_dimension)

# Build model
optimizer = optimizers.Adam(lr=lr)
model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Model summary
print_title("Model Summary")
model.summary()

#################################################################
# IV. Train the Model
#################################################################
print('')
print_title("Start Training")

#================================================================
# Callbacks
#================================================================
callbacks = []
# ModelCheckpoint 
ckpt = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(res_root_dir, res_folder_name, 'best_model.hdf5'), 
                monitor='val_loss', 
                verbose=1, 
                save_best_only=True,
                mode='auto')
callbacks.append(ckpt)
# EarlyStopping
earlyStopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience= early_stopping_patience,
                verbose=1,
                mode='auto')
callbacks.append(earlyStopping)
# Tensorboard
tb_log_dir = os.path.join(res_root_dir, res_folder_name, 'logs')
#if not os.path.exists(tb_log_dir):
#    os.makedirs(tb_log_dir)
tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=tb_log_dir
                )
callbacks.append(tensorboard)
#ReduceLROnPlateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                verbose=1,
                                factor=0.5,
                                patience=reduce_lr_patience)
                                
callbacks.append(reduce_lr)

# Training loop with generators
H = model.fit_generator(
        train_gen,
        steps_per_epoch=np.ceil(german_data.train_size/german_data.batch_size),
        epochs=epochs,
        callbacks = callbacks,
        validation_data=val_gen,
        validation_steps=np.ceil(german_data.val_size/german_data.batch_size),
        #class_weight=german_data.class_weight
        )

print('')
print("[KF INFO] Training Completed!")
print('')

#################################################################
# V. Predict
#################################################################
from tensorflow.keras.models import load_model

print_title("Predicting with round 1 test data")

test_data = german_data.getTestData()
# predicting process
#res = model.predict(test_data)
# Load best model
model_best = load_model(os.path.join(res_root_dir, res_folder_name, 'best_model.hdf5'))
res = model_best.predict(test_data)
# find the largest confident'test_concats index
res = np.argmax(res, axis=1)

# Convert categorical result to one-hot encoded matrix
final_res = tf.keras.utils.to_categorical(res)
final_res = final_res.astype(int)
print("[KF INFO] Final prediction result:", final_res)
print("[KF INFO] Prediction shape:", final_res.shape)

# Save prediction to CSV

csv_name = 'prediction-%d%02d%02d-%s-epochs-%d-trainsize-%d.csv' % (cur_date.year, cur_date.month, cur_date.day, model_name, epochs, german_data.train_size)
pred_dir = 'predictions'

np.savetxt(os.path.join(pred_dir, csv_name), final_res, fmt='%d', delimiter=',')
np.savetxt(os.path.join(res_root_dir, res_folder_name, csv_name), final_res, fmt='%d', delimiter=',')
print('[KF INFO] Prediction csv saved!')

#################################################################
# VI. Save Plot
#################################################################
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print_title("Plot the Loss and Accuracy")
# plot the loss and accuracy
plt.style.use("ggplot")
plt.figure()
#N = epochs
N = len(H.history["loss"])
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(os.path.join(res_root_dir, res_folder_name, 'plot.png'))

