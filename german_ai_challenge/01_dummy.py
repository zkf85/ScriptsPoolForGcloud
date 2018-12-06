############################################################
# Alibaba Cloud German AI Challenge 2018
# KF 2018/12/06
############################################################
# I. Imports and Paths
import os
import numpy as np
import h5py
import tensorflow as tf
import csv

# Check current working directory
os.getcwd()

# Set paths
base_dir = os.path.expanduser('/tmp/german_dataset')
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_round1_test = os.path.join(base_dir,'round1_test_a_20181109.h5')

# Validate the paths
print('')
print('='*60)
print('Paths')
print('='*60)
print("Current directory : ", os.listdir(base_dir))
print('')

# II. Overview of the Data
fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
fid_test1 = h5py.File(path_round1_test, 'r')
# Have a look at keys stored in the h5 files
print('='*60)
print("Overview of Data:")
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

# III. Simply Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(17, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])


# IV. Train Data Flow
batch_size = 10000

for i in range(0, 10000, batch_size):
    if i % 10 == 0:
        print("[KF INFO] Data flow - loaded %d/350000" % i)
    start_pos = i
    end_pos = i + batch_size
    train_s1_x_batch = np.asarray(s1_training[start_pos:end_pos])
    train_s1_y_batch = np.asarray(label_training[start_pos:end_pos])
    model.fit(train_s1_x_batch, train_s1_y_batch, epochs=1)

print(model.evaluate(s1_validation[:batch_size], label_validation[:batch_size]))

# V. Predict
res = model.predict(s1_test1)
res = np.argmax(res, axis=1)
print(res)
print(len(res))

final_res = tf.keras.utils.to_categorical(res)
final_res = final_res.astype(int)
print(final_res)
print(len(final_res))

with open('submission.csv', 'w') as f:
    writer = csv.writer(f)
    for line in final_res:
        writer.writerow(line)
    print('submission csv saved!')
