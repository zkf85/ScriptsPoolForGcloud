############################################################
# Alibaba Cloud German AI Challenge 2018
# KF 2018/12/06
#    2018/12/07
############################################################

############################################################
# I. Imports and Paths
############################################################
import os
import numpy as np
import h5py
import tensorflow as tf
import csv


# Set paths
base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')

path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_round1_test = os.path.join(base_dir,'round1_test_a_20181109.h5')

# Validate the paths
print('')
print('='*60)
print(' '*20, 'Paths')
print('='*60)
print("Validate data in the base directory : ")
print(os.listdir(base_dir))
print('')

############################################################
# II. Overview of the Data
############################################################
fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
fid_test1 = h5py.File(path_round1_test, 'r')

# Have a look at keys stored in the h5 files
print('')
print('='*60)
print(' '*20, "Overview of Data:")
print('='*60)
print('Training data keys   :', list(fid_training.keys()))
print('Validation data keys :', list(fid_validation.keys()))
print('Round1 Test data keys :', list(fid_test1.keys()))

print('-'*60)
print("Training data shapes:")
s1_training = fid_training['sen1']
s2_training = fid_training['sen2']
label_training = fid_training['label']
print('  Sentinel 1 data shape :', s1_training.shape)
print('  Sentinel 2 data shape :', s2_training.shape)
print('  Label data shape      :', label_training.shape)
print('-'*60)
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

# III. Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(17, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

model.summary()

############################################################
# IV. Train Data Flow
############################################################

# Set flag for test mode or not
#is_test = True
is_test = False

# Set real epoch for the training process
epochs = 3

# Set data flow block size 
block_size = pow(2, 13)   

# Set batch_size 
batch_size = 64

if is_test:
    train_size = 30000
    val_size = 10000

else:
    train_size = len(label_training)
    val_size = len(label_validation)

print('')
print('')
print('='*60)
print(' '*20, "Start Training")
print('='*60)

# Training loop
for e in range(epochs):
    print("")
    print("-"*60)
    print("[KF INFO] EPOCH :", e + 1)
    print("-"*60)

    for i in range(0, train_size, block_size):
        if i % 10 == 0:
            print("[KF INFO] EPOCH %d >> data trained : %d/%d" % (e + 1, i, train_size))
        start_pos = i
        end_pos = min(i + block_size, train_size)
        train_s1_x_block = np.asarray(s1_training[start_pos:end_pos])
        train_s2_x_block = np.asarray(s2_training[start_pos:end_pos])
        train_y_block = np.asarray(label_training[start_pos:end_pos])
        train_concat_x_block = np.concatenate([train_s1_x_block, train_s2_x_block], axis=-1) 

        #print(train_concat_x_block.shape, train_y_block.shape)
        model.fit(train_concat_x_block, train_y_block, epochs=1, batch_size=batch_size)

    # Validate for each epoch
    print('')
    print("[KF INFO] Validating :")
    loss, acc_num = 0.0, 0.0
    # Validation loop
    for i in range(0, val_size, block_size): 
        
        start_pos = i
        end_pos = min(i + block_size, val_size)
        val_s1_x_block = np.asarray(s1_validation[start_pos:end_pos])
        val_s2_x_block = np.asarray(s2_validation[start_pos:end_pos])
        val_y_block = np.asarray(label_validation[start_pos:end_pos])
        
        val_concat_x_block = np.concatenate([val_s1_x_block, val_s2_x_block], axis=-1) 
        #print(val_concat_x_block.shape, train_y_block.shape)

        val_res = model.evaluate(val_concat_x_block, val_y_block, batch_size=batch_size)
        # Cumulate the loss and accuracy
        loss += val_res[0]*(end_pos - i)
        acc_num += val_res[1]*(end_pos - i)

    print("[KF INFO] val loss : %.4f, val acc : %.4f" % (loss / val_size, acc_num / val_size))

print('')
print("[KF INFO] Training Completed!")
print('')

# V. Predict
print('')
print('='*60)
print(' '*10, "Predicting with round 1 test data")
print('='*60)
test_concat = np.concatenate([s1_test1, s2_test1], axis=-1)
res = model.predict(test_concat)
res = np.argmax(res, axis=1)

# Convert categorical result to one-hot encoded matrix
final_res = tf.keras.utils.to_categorical(res)
final_res = final_res.astype(int)
print("[KF INFO] Final prediction result:", final_res)
print("[KF INFO] Prediction shape:", final_res.shape)

# Save prediction to CSV
from datetime import datetime

cur_date = datetime.now()
csv_name = 'prediction-%d%d%d-epochs-%d-trainsize-%d' % (cur_date.year, cur_date.month, cur_date.day, epochs, train_size)
pred_dir = 'predictions'

with open(os.path.join(pred_dir, csv_name), 'w') as f:
    writer = csv.writer(f)
    for line in final_res:
        writer.writerow(line)
    print('[KF INFO] Prediction csv saved!')
