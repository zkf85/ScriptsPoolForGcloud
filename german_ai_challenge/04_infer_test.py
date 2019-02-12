# KF 01/07/2019
# KF 01/22/2019 - update for round2_testA
# 
# Use given model to generate submission for testB
#
import os

# To Force using CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
#result_sub_dir = 'model-KFSmallerVGGNet-20181227-epochs-100-trainsize-19296-channels-s2'
#result_sub_dir = 'model-20190114-KFSmallerVGGNet-epochs-100-trainsize-352366-channels-s1_ch5678+s2'
#result_sub_dir = 'model-20190123-KFResNet34-epochs-100-trainsize-352366-channels-s2'
#result_sub_dir = 'model-KFResNet18-20190104-epochs-200-trainsize-19296-channels-s2'
result_sub_dir = 'model-20190129-KFSmallerVGGNet-epochs-100-trainsize-19296-channels-s1_ch5678+s2'

result_dir = os.path.join(result_base_dir, result_sub_dir)

# Channel:
#channel = 'full'
channel = 's1_ch5678+s2'
#channel = 's2'

#===============================================================================
# Load test data based on chosen channel
#===============================================================================
#data_base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')
#round1_testB_filename = 'round1_test_b_20190104.h5'
##round2_testA_filename = 'round2_test_a_20190121.h5'
#
#fid_test1B = h5py.File(os.path.join(data_base_dir, round1_testB_filename), 'r')
#
#s1_test1B = fid_test1B['sen1']
#s2_test1B = fid_test1B['sen2']
#
## Choose Channel
#if channel == 'full': 
#    test_data = np.concatenate([s1_test1B, s2_test1B], axis=-1)
#
#elif channel == 's2_rgb':
#    tmp = []
#    for p in s2_test1B:
#        tmp.append(p[...,2::-1])
#    test_data = np.asarray(tmp)
#
#elif channel == 's1': 
#    test_data = s1_test1B
#
#elif channel == 's2': 
#    test_data = s2_test1B
#
#elif channel == 's1_ch5678': 
#    test_data = s1_test1B[...,4:]
#
##===============================================================================
## Load Model and Predict
##===============================================================================
## Load best model
#model_best = load_model(os.path.join(result_dir, 'best_model.hdf5'))
#res = model_best.predict(test_data)
## find the largest confident'test_concats index
#res = np.argmax(res, axis=1)
#
## Convert categorical result to one-hot encoded matrix
#final_res = tf.keras.utils.to_categorical(res)
#final_res = final_res.astype(int)
#print("[KF INFO] Final prediction result for Round1 testB:", final_res)
#print("[KF INFO] Prediction shape:", final_res.shape)
#
## Save prediction to CSV
#
#csv_name = 'prediction-testB-' + result_sub_dir + '.csv'
#pred_dir = 'predictions'
#
#np.savetxt(os.path.join(pred_dir, csv_name), final_res, fmt='%d', delimiter=',')
#np.savetxt(os.path.join(result_dir, csv_name), final_res, fmt='%d', delimiter=',')
#print('[KF INFO] Prediction csv saved!')

#===============================================================================
# Load test data using GermanData Class
# KF 01/29/2019 Update
#===============================================================================
base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')
train_filename = 'training.h5'
val_filename = 'validation.h5'
round1_testA_filename = 'round1_test_a_20181109.h5'
round1_testB_filename = 'round1_test_b_20190104.h5'
round2_testA_filename = 'round2_test_a_20190121.h5'

# Parameters
train_mode = 'real'
batch_size = 32
#data_channel = 's1_ch5678+s2'
data_channel = 's2'
#data_gen_mode = 'val_dataset_only'
data_gen_mode = 'kf_data_only'
data_normalize = 'no'

from kfdata.KFGermanData import GermanData
# Parameter dictionary for initializing GermanData instance
param_dict = {}
param_dict['base_dir'] = base_dir
param_dict['train_filename'] = train_filename
param_dict['val_filename'] = val_filename
param_dict['round1_testA_filename'] = round1_testA_filename
param_dict['round1_testB_filename'] = round1_testB_filename
param_dict['round2_testA_filename'] = round2_testA_filename

kf_test2B_filename = 'kf_test2B_3sigma_standardized.h5'
param_dict['kf_test2B_filename'] = kf_test2B_filename

param_dict['train_mode'] = train_mode
param_dict['batch_size'] = batch_size
param_dict['data_channel'] = data_channel
param_dict['data_gen_mode'] = data_gen_mode
param_dict['data_normalize'] = data_normalize

# Create GermanData class instance
german_data = GermanData(param_dict)

from tensorflow.keras.models import load_model

#print_title("Predicting with round 1 test data")
#print("Predicting with round 2 test A data")
print("Predicting with round 2 test B data")

#test_data = german_data.getTest1AData()
#test_data = german_data.getTest1BData()
#test_data = german_data.getTest2AData()
test_data = german_data.getTest2BData()

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
print("[KF INFO] Final prediction result:", final_res)
print("[KF INFO] Prediction shape:", final_res.shape)

# Save prediction to CSV

#csv_name = 'prediction-test2A-' + result_sub_dir + '.csv'
csv_name = 'prediction-test2B-' + result_sub_dir + '.csv'
pred_dir = 'predictions'

np.savetxt(os.path.join(pred_dir, csv_name), final_res, fmt='%d', delimiter=',')
np.savetxt(os.path.join(result_dir, csv_name), final_res, fmt='%d', delimiter=',')
print('[KF INFO] Prediction csv saved!')
