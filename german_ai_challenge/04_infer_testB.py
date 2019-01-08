# KF 01/07/2019
# 
# Use given model to generate submission for testB
#
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import csv

#===============================================================================
# Load Model
#===============================================================================
# Path:
result_base_dir = '/home/kefeng/German_AI_Challenge/results'
result_sub_dir = 'model-KFSmallerVGGNet-20181227-epochs-100-trainsize-19296-channels-s2'
result_dir = os.path.join(result_base_dir, result_sub_dir)

# Channel:
#channel = 'full'
channel = 's2'

#===============================================================================
# Load testB data based on chosen channel
#===============================================================================
data_base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')
round1_testB_filename = 'round1_test_b_20190104.h5'

fid_test1B = h5py.File(os.path.join(data_base_dir, round1_testB_filename), 'r')

s1_test1B = fid_test1B['sen1']
s2_test1B = fid_test1B['sen2']

# Choose Channel
if channel == 'full': 
    test_data = np.concatenate([s1_test1B, s2_test1B], axis=-1)

elif channel == 's2_rgb':
    tmp = []
    for p in s2_test1B:
        tmp.append(p[...,2::-1])
    test_data = np.asarray(tmp)

elif channel == 's1': 
    test_data = s1_test1B

elif channel == 's2': 
    test_data = s2_test1B

elif channel == 's1_ch5678': 
    test_data = s1_test1B[...,4:]

#===============================================================================
# Load Model and Predict
#===============================================================================
# Load best model
model_best = load_model(os.path.join(result_dir, 'best_model.hdf5'))
res = model_best.predict(test_data)
# find the largest confident'test_concats index
res = np.argmax(res, axis=1)

# Convert categorical result to one-hot encoded matrix
final_res = tf.keras.utils.to_categorical(res)
final_res = final_res.astype(int)
print("[KF INFO] Final prediction result for Round1 testB:", final_res)
print("[KF INFO] Prediction shape:", final_res.shape)

# Save prediction to CSV

csv_name = 'prediction-testB-' + result_sub_dir + '.csv'
pred_dir = 'predictions'

np.savetxt(os.path.join(pred_dir, csv_name), final_res, fmt='%d', delimiter=',')
np.savetxt(os.path.join(result_dir, csv_name), final_res, fmt='%d', delimiter=',')
print('[KF INFO] Prediction csv saved!')
