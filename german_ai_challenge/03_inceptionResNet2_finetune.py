# 02_smallvgg_balanced_generator.py
#################################################################
# Alibaba Cloud German AI Challenge 2018
# KF 2018/12/06
#    2018/12/07
#    2018/12/10 - add generators
#    2018/12/17 - make a balanced generator
#################################################################
import os
import random
import numpy as np
import h5py
import tensorflow as tf
from datetime import datetime
import csv

# Load current date for later use
cur_date = datetime.now()

# Print title with double lines with text aligned to center
def print_title(title):
    print('')
    print('=' * 65)
    print(' ' * ((65 - len(title))//2 - 1), title)
    print('=' * 65)

#################################################################
# I. Load Data
#################################################################
#----------------------------------------------------------------
# Set Paths
#----------------------------------------------------------------
base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')
train_filename = 'training.h5'
val_filename = 'validation.h5'
round1_test_filename = 'round1_test_a_20181109.h5'

# Package parameters into dictionary
param_dict = {}
param_dict['base_dir'] = base_dir
param_dict['train_filename'] = train_filename
param_dict['val_filename'] = val_filename
param_dict['round1_test_filename'] = round1_test_filename

#----------------------------------------------------------------
# Load data and show data info
#----------------------------------------------------------------
from kfdata.KFGermanData import GermanData

german_data = GermanData(param_dict)


#################################################################
# I. Set Parameters
#################################################################
#----------------------------------------------------------------
# Set Train mode: either 'real' or 'test'
#----------------------------------------------------------------
#train_mode = 'real'
train_mode = 'test'

#----------------------------------------------------------------
# Set real epoch for the training process
#----------------------------------------------------------------
epochs = 25

#----------------------------------------------------------------
# Set batch_size 
#----------------------------------------------------------------
batch_size = 64

#----------------------------------------------------------------
# Set data channel:
#----------------------------------------------------------------
#data_channel = 'full'
data_channel = 's2_rgb'

#----------------------------------------------------------------
# Set data generating mode: either 'original' or 'balanced'
#----------------------------------------------------------------
#gen_mode = 'original'
data_gen_mode = 'balanced'

#----------------------------------------------------------------
# Set model name
#----------------------------------------------------------------
model_name = 'KFSmallerVGGNet'

#----------------------------------------------------------------
# Get Train/Val size
#----------------------------------------------------------------
if gen_mode =='original':
    train_size = german_data.get_train_size(train_mode)
    val_size = german_data.get_val_size(train_mode)
elif gen_mode == 'balanced':
    train_size = german_data.get_balanced_train_size(train_mode)
    val_size = german_data.get_balanced_val_size(train_mode)

#----------------------------------------------------------------
# Set model saving Path 
# Requires: model_name, cur_date, epochs, train_size
#----------------------------------------------------------------
res_root_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/results')
res_folder_name = 'model-%s-%d%d%d-epochs-%d-trainsize-%d' % (model_name, cur_date.year, cur_date.month, cur_date.day, epochs, train_size)
if not os.path.exists(os.path.join(res_root_dir, res_folder_name)):
    os.makedirs(os.path.join(res_root_dir, res_folder_name))

#################################################################
# Print Training Parameters
#################################################################
print('')
print_title("Training Parameters")
print("Train Mode       :", train_mode)
print("Data Channel     :", data_channel)
print("Data Gen Mode    :", data_gen_mode)
print("Train Size       :", train_size)
print("Validation Size  :", val_size)
print("Epochs           :", epochs)
print("Batch Size       :", )
print('-'*65)
print("Model Saving Directory:")
print(os.path.join(res_root_dir, res_folder_name))
print('-'*65)

print('')
print_title("Start Training")

#################################################################
# III. Build Model
#################################################################

from kfmodels.kfsmallervggnet import KFSmallerVGGNet
# model name for saving directory naming
model = KFSmallerVGGNet.build(german_data.getDataShape(channel=data_channel))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print_title("Model Summary")
model.summary()

#################################################################
# IV. Train with Data Generator 
#################################################################
# Callbacks
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
                patience=10,
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

# Training loop with generators
H = model.fit_generator(
        trainGenerator(batch_size),
        steps_per_epoch=np.ceil(train_size/batch_size),
        epochs=epochs,
        callbacks = callbacks,
        validation_data=valGenerator(batch_size),
        validation_steps=np.ceil(val_size/batch_size)
        )

print('')
print("[KF INFO] Training Completed!")
print('')

#################################################################
# V. Predict
#################################################################
print_title("Predicting with round 1 test data")

# concatenate s1 and s2 data along the last axis
test_concat = np.concatenate([s1_test1, s2_test1], axis=-1)
# predicting process
res = model.predict(test_concat)
# find the largest confident's index
res = np.argmax(res, axis=1)

# Convert categorical result to one-hot encoded matrix
final_res = tf.keras.utils.to_categorical(res)
final_res = final_res.astype(int)
print("[KF INFO] Final prediction result:", final_res)
print("[KF INFO] Prediction shape:", final_res.shape)

# Save prediction to CSV

csv_name = 'prediction-%s-%d%d%d-epochs-%d-trainsize-%d.csv' % (model_name, cur_date.year, cur_date.month, cur_date.day, epochs, train_size)
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
N = len(H.history["loss"]
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(os.path.join(res_root_dir, res_folder_name, 'plot.png'))
