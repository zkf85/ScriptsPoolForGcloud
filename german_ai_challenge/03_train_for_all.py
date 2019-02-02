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
#    2019/01/14 - upgrade the data generators with data_gen_mode "shuffled_original"
#               - add channel option "s1_ch5678+s2"                
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
round1_testA_filename = 'round1_test_a_20181109.h5'
round1_testB_filename = 'round1_test_b_20190104.h5'
round2_testA_filename = 'round2_test_a_20190121.h5'
#round2_testB_filename = 'round2_test_b_20190104.h5'

# KF 01/31/2019
#kf_data_filename = 'kf_data_shuffled.h5'
#kf_val_filename = 'kf_val_10k.h5'
#kf_test2A_filename = 'round2_test_a_20190121.h5'

# KF 02/02/2019
kf_data_filename = 'kf_data_shuffled_3sigma_standardized.h5'
kf_val_filename = 'kf_val_10k_3sigma_standardized.h5'
kf_test2A_filename = 'kf_test2A_3sigma_standardized.h5'

# Set Train mode: 'real' or 'test'
train_mode = 'real'
#train_mode = 'test'

# Set real epoch for the training process
#epochs = 3
epochs = 100 
#epochs = 200

# Set batch_size 
#batch_size = 256
#batch_size = 128
batch_size = 64
#batch_size = 32
#batch_size = 16
#batch_size = 8
#batch_size = 4

# Initial learning rate 
#lr = 0.001
#lr = 0.0003
lr = 0.0001
#lr = 3e-5
#lr = 1e-5

# Early Stopping patience:
#early_stopping_patience = 10
#early_stopping_patience = 20
early_stopping_patience = 30

# ReduceLRPlateau patience:
#reduce_lr_patience = 6
#reduce_lr_patience = 8
reduce_lr_patience = 10

# Set data channel: 
#data_channel = 'full'
#data_channel = 's2_rgb'
#data_channel = 's1'
data_channel = 's2'
#data_channel = 's1_ch5678'
#data_channel = 's1_ch5678+s2'

# Set data generating mode: 
# if original, class_weight should be set
#data_gen_mode = 'original'
#data_gen_mode = 'shuffled_original'
#data_gen_mode = 'balanced'
#data_gen_mode = 'val_dataset_only'
data_gen_mode = 'kf'
#data_gen_mode = 'kf_data_only'

# Data normalize or not
#data_normalize = 'yes'
data_normalize = 'no'

# Set model name
model_name = 'KFSmallerVGGNet'
#model_name = 'KFDummy'
#model_name = 'KFResNet18'
#model_name = 'KFResNet34'
#model_name = 'KFResNet50'
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
param_dict['kf_data_filename'] = kf_data_filename
param_dict['kf_val_filename'] = kf_val_filename
param_dict['kf_test2A_filename'] = kf_test2A_filename
#param_dict['kf_test2B_filename'] = kf_test2B_filename

param_dict['round1_testA_filename'] = round1_testA_filename
param_dict['round1_testB_filename'] = round1_testB_filename
param_dict['round2_testA_filename'] = round2_testA_filename
#param_dict['round2_testB_filename'] = round2_testB_filename

param_dict['train_mode'] = train_mode
param_dict['batch_size'] = batch_size
param_dict['data_channel'] = data_channel
param_dict['data_gen_mode'] = data_gen_mode
param_dict['data_normalize'] = data_normalize

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

# KF 01/22/2019
# Save the History to a file
print('H.history type:', type(H.history))
print('H.history keys:', H.history.keys())

import pickle
hist_file_path = os.path.join(res_root_dir, res_folder_name, 'history.pkl')

with open(hist_file_path, 'wb') as f:
    pickle.dump(H.history, f)

#################################################################
# V. Predict
#################################################################
from tensorflow.keras.models import load_model

#print_title("Predicting with round 1 test data")
print_title("Predicting with round 2 test A data")
#print_title("Predicting with round 2 test B data")

#test_data = german_data.getTest1AData()
#test_data = german_data.getTest1BData()
test_data = german_data.getTest2AData()
#test_data = german_data.getTest2BData()
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
csv_name = 'prediction-test2A-%d%02d%02d-%s-epochs-%d-trainsize-%d-channels-%s.csv' % (cur_date.year, cur_date.month, cur_date.day, model_name, epochs, german_data.train_size, data_channel)
#csv_name = 'prediction-test2B-%d%02d%02d-%s-epochs-%d-trainsize-%d-channels-%s.csv' % (cur_date.year, cur_date.month, cur_date.day, model_name, epochs, german_data.train_size, data_channel)
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
N = len(H.history["loss"])
#plt.style.use("ggplot")
#plt.figure()
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()
#N = epochs
l1 = ax1.plot(np.arange(0, N), H.history["acc"], label="train_acc")
l2 = ax1.plot(np.arange(0, N), H.history["val_acc"], label="val_acc", linewidth=2)
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch #')
ax1.set_ylim(0, 1.05)
ax1.grid()

l3 = ax2.plot(np.arange(0, N), H.history["loss"], color='orchid', linestyle='dashed', label="train_loss")
l4 = ax2.plot(np.arange(0, N), H.history["val_loss"], color='limegreen', linestyle='dashed', label="val_loss", linewidth=2)
ax2.set_ylabel('Loss')
# Put all label legend together
l = l1 + l2 + l3 + l4
labels = [i.get_label() for i in l]
#plt.legend(l, labels, loc='center right')
plt.legend(l, labels, loc='center right')

plt.title("Training Loss and Accuracy")
plt_name = 'plt-acc-loss-%d%02d%02d-%s.eps' % (cur_date.year, cur_date.month, cur_date.day, model_name)
plt.savefig(os.path.join(res_root_dir, res_folder_name, plt_name), format='eps', dpi=1000)

#################################################################
# Plot Learning Rate`
#################################################################
print_title("Plot Learning Rate")
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, N), H.history['lr'], linewidth=4)
plt.xlabel('Epoch #')
plt.ylabel('Learning Rate')
plt.grid()

plt.title("Learning Rate")
plt_name = 'plt-lr-%d%02d%02d-%s.eps' % (cur_date.year, cur_date.month, cur_date.day, model_name)
plt.savefig(os.path.join(res_root_dir, res_folder_name, plt_name), format='eps', dpi=1000)
