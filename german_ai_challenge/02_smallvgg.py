# 01_dummy.py
#################################################################
# Alibaba Cloud German AI Challenge 2018
# KF 2018/12/06
#    2018/12/07
#    2018/12/10 - add generators
#################################################################
import os
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
# I. Paths
#################################################################
# Set paths
base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_round1_test = os.path.join(base_dir,'round1_test_a_20181109.h5')

# Validate the paths
print_title('Paths')
print("Validate data in the base directory : ")
print(os.listdir(base_dir))
print('')

#################################################################
# II. Overview of the Data
#################################################################
fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
fid_test1 = h5py.File(path_round1_test, 'r')

# Have a look at keys stored in the h5 files
print_title("Overview of Data:")
print('Training data keys   :', list(fid_training.keys()))
print('Validation data keys :', list(fid_validation.keys()))
print('Round1 Test data keys :', list(fid_test1.keys()))

print('-'*65)
print("Training data shapes:")
s1_training = fid_training['sen1']
s2_training = fid_training['sen2']
label_training = fid_training['label']
print('  Sentinel 1 data shape :', s1_training.shape)
print('  Sentinel 2 data shape :', s2_training.shape)
print('  Label data shape      :', label_training.shape)
print('-'*65)
print("Validation data shapes:")
s1_validation = fid_validation['sen1']
s2_validation = fid_validation['sen2']
label_validation = fid_validation['label']
print('  Sentinel 1 data shape :', s1_validation.shape)
print('  Sentinel 2 data shape :', s2_validation.shape)
print('  Label data shape      :', label_validation.shape)
print("Round1 Test data shapes:")
s1_test1 = fid_test1['sen1']
s2_test1 = fid_test1['sen2']
print('  Sentinel 1 data shape :', s1_test1.shape)
print('  Sentinel 2 data shape :', s2_test1.shape)

# Save input dimension parameters
input_width = s1_training.shape[1]
input_height = s1_training.shape[2]
s1_channel = s1_training.shape[3]
s2_channel = s2_training.shape[3]
label_dim = label_training.shape[1]

#################################################################
# III. Build Model
#################################################################

# KFSmallerVGGNet
from kfmodels.kfsmallervggnet import KFSmallerVGGNet
model_name = 'KFSmallerVGGNet'
model = KFSmallerVGGNet.build(input_width, 
                                input_height, 
                                s1_channel + s2_channel,
                                label_dim)
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

print_title("Model Summary")
model.summary()

#################################################################
# IV. Train with Data Generator 
#################################################################
# Train Mode
train_mode = 'Full'

# Set flag for test mode or no
#is_test = True
is_test = False

# Set real epoch for the training process
epochs = 10

# Set batch_size 
batch_size = 128

# training size / validation size
train_size = len(label_training)
val_size = len(label_validation)

# Test parameters
if is_test:
    train_mode = 'Test'
    train_size = 30000
    val_size = 10000
    #epochs = 3    
    batch_size = 32

# Model saving Path
res_root_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/results')
res_folder_name = 'model-%s-%d%d%d-epochs-%d-trainsize-%d' % (model_name, cur_date.year, cur_date.month, cur_date.day, epochs, train_size)
if not os.path.exists(os.path.join(res_root_dir, res_folder_name)):
    os.makedirs(os.path.join(res_root_dir, res_folder_name))

# List Training Parameters
print('')
print_title("Training Parameters")
print("Model Saving Directory:")
print(os.path.join(res_root_dir, res_folder_name))
print('-'*65)
print("Test Mode        :", train_mode)
print("Train Size       :", train_size)
print("Validation Size  :", val_size)
print("Epochs           :", epochs)
print("Batch Size       :", batch_size)
print('-'*65)

print('')
print_title("Start Training")

# Create Training Generator
def trainGenerator(batch_size):
    # Generate data with batch_size 
    while True:
        for i in range(0, train_size, batch_size):
            start_pos = i
            end_pos = min(i + batch_size, train_size)
            train_s1_X_batch = np.asarray(s1_training[start_pos:end_pos])
            train_s2_X_batch = np.asarray(s2_training[start_pos:end_pos])
            train_y_batch = np.asarray(label_training[start_pos:end_pos])
            # concatenate s1 and s2 data along the last axis
            train_concat_X_batch = np.concatenate([train_s1_X_batch, train_s2_X_batch], axis=-1) 
            # According to "fit_generator" on Keras.io, the output from the generator must
            # be a tuple (inputs, targets), thus,
            yield (train_concat_X_batch, train_y_batch)

# Create Valication Generator
def valGenerator(batch_size):
    while True:
        # Generate data with batch_size 
        for i in range(0, val_size, batch_size):
            start_pos = i
            end_pos = min(i + batch_size, val_size)
            val_s1_X_batch = np.asarray(s1_validation[start_pos:end_pos])
            val_s2_X_batch = np.asarray(s2_validation[start_pos:end_pos])
            val_y_batch = np.asarray(label_validation[start_pos:end_pos])
            # concatenate s1 and s2 data along the last axis
            val_concat_X_batch = np.concatenate([val_s1_X_batch, val_s2_X_batch], axis=-1) 
            # According to "fit_generator" on Keras.io, the output from the generator must
            # be a tuple (inputs, targets), thus,
            yield (val_concat_X_batch, val_y_batch)

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
                patience=3,
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
# V. Predict and Save
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

# plot the loss and accuracy
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
plt.savefig(os.path.join(res_root_dir, res_folder_name, 'plot.png'))
