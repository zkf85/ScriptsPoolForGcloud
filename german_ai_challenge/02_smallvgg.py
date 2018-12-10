# 02_smallvgg.py
#################################################################
# Alibaba Cloud German AI Challenge 2018
# KF 2018/12/06
#    2018/12/07
#################################################################
import os
import numpy as np
import h5py
import tensorflow as tf
import csv

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
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

# Build small vgg model from scratch
model = Sequential()
input_shape = (input_width, input_height, s1_channel + s2_channel)
chanDim = -1
# CONV => RELU => POOL
model.add(Conv2D(128, (3, 3), padding="same",
        input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(2048))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(label_dim))
model.add(Activation("softmax"))

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

print_title("Model Summary")
model.summary()

#################################################################
# IV. Train Data Flow
#################################################################
# Set flag for test mode or not
is_test = True
#is_test = False

# Set real epoch for the training process
epochs = 3

# Set data flow block size 
block_size = pow(2, 13)   

# Set batch_size 
batch_size =64 

if is_test:
    train_size = 30000
    val_size = 10000
    epochs = 3    
    block_size = 4096
    batch_size = 64

else:
    train_size = len(label_training)
    val_size = len(label_validation)

print('')
print_title("Start Training")

# Training loop
for e in range(epochs):
    print("")
    print("-"*65)
    print("[KF INFO] EPOCH :", e + 1)
    print("-"*65)

    for i in range(0, train_size, block_size):
        if i % (5*block_size) == 0:
            print("[KF INFO] EPOCH %d >> data trained : %d/%d" % (e + 1, i, train_size))
        start_pos = i
        end_pos = min(i + block_size, train_size)
        train_s1_x_block = np.asarray(s1_training[start_pos:end_pos])
        train_s2_x_block = np.asarray(s2_training[start_pos:end_pos])
        train_y_block = np.asarray(label_training[start_pos:end_pos])
        # concatenate s1 and s2 data along the last axis
        train_concat_x_block = np.concatenate([train_s1_x_block, train_s2_x_block], axis=-1) 
        # training process
        model.fit(train_concat_x_block, train_y_block, epochs=1, batch_size=batch_size)

    # Validate for each epoch
    print('')
    print("[KF INFO] Validating :")

    # Validation loop
    loss, acc_num = 0.0, 0.0
    for i in range(0, val_size, block_size): 
        
        start_pos = i
        end_pos = min(i + block_size, val_size)
        val_s1_x_block = np.asarray(s1_validation[start_pos:end_pos])
        val_s2_x_block = np.asarray(s2_validation[start_pos:end_pos])
        val_y_block = np.asarray(label_validation[start_pos:end_pos])
        # concatenate s1 and s2 data along the last axis
        val_concat_x_block = np.concatenate([val_s1_x_block, val_s2_x_block], axis=-1) 
        # validation process
        val_res = model.evaluate(val_concat_x_block, val_y_block, batch_size=batch_size)
        # cumulate the loss and accuracy
        loss += val_res[0]*(end_pos - i)
        acc_num += val_res[1]*(end_pos - i)

    print("[KF INFO] val_loss : %.4f, val_acc : %.4f" % (loss / val_size, acc_num / val_size))

    # Try this:
    #val_s1 = np.asarray(s1_training[-5000:])
    #val_s2 = np.asarray(s2_training[-5000:])
    #val_y = np.asarray(label_training[-5000:])
    #val_res = model.evaluate(np.concatenate([val_s1, val_s2], axis=-1), val_y, batch_size=batch_size)
    #print("[KF INFO] val_loss : %.4f, val_acc : %.4f" % (val_res[0], val_res[1]))

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
from datetime import datetime

cur_date = datetime.now()
csv_name = 'prediction-%d%d%d-epochs-%d-trainsize-%d.csv' % (cur_date.year, cur_date.month, cur_date.day, epochs, train_size)
pred_dir = 'predictions'

np.savetxt(os.path.join(pred_dir, csv_name), final_res, fmt='%d', delimiter=',')
print('[KF INFO] Prediction csv saved!')
