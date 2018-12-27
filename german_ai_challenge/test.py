# test.py
#################################################################
# Created by 
#   KF 12/18/2018
# 
# Updated:
#   KF 12/26/2018
#
# Re-construct the code for Ali-German AI Challenger AI Competition
# Distribute functions into modules.
#
#################################################################
import os
import numpy as np

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

#param_dict['train_mode'] = 'test'
param_dict['train_mode'] = 'real'
#param_dict['batch_size'] = 64
param_dict['batch_size'] = 128
param_dict['data_channel'] = 'full'
#param_dict['data_channel'] = 's2_rgb'
#param_dict['data_gen_mode'] = 'original'
#param_dict['data_gen_mode'] = 'balanced'
param_dict['data_gen_mode'] = 'val_dataset_only'

#================================================================
# Test Initialization 
#================================================================
from kfdata.KFGermanData import GermanData

german_data = GermanData(param_dict)

#----------------------------------------------------------------
# Print Training Parameters
#----------------------------------------------------------------
print('')
german_data.print_title("Training Parameters")
print("Train Mode       :", german_data.train_mode)
print("Data Channel     :", german_data.data_channel)
print("Data Gen Mode    :", german_data.data_gen_mode)
print("Train Size       :", german_data.train_size)
print("Validation Size  :", german_data.val_size)
print("Batch Size       :", german_data.batch_size)
print("Data Dimension   :", german_data.data_dimension)
print('-'*65)

#================================================================
# Test -> Generators:
# KF 12/18/2018
#================================================================
loop = 10
#print('')
#print('[KF INFO] Test german_data.train_gen:')
#print('-'*65)
#for i in range(loop):
#    data = next(german_data.train_gen)
#    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)
#
#print('')
#print('[KF INFO] Test german_data.val_gen:')
#print('-'*65)
#for i in range(loop):
#    data = next(german_data.val_gen)
#    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

#================================================================
# Test -> getTestData
# KF 12/18/2018
#================================================================
#test_data = german_data.getTestData()
#print(test_data.min())

#================================================================
# Test -> label_qty
#================================================================
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#
#plt.plot(german_data.label_qty)
#plt.savefig('label_qty.png')
#
#print("class_weight :")
#print(german_data.class_weight)
#plt.figure()
#plt.plot(german_data.class_weight)
#plt.savefig('class_weight.png')

#================================================================
# Test -> Generators with validation dataset ONLY:
# KF 12/26/2018
#================================================================
loop = 10
print('')
print('[KF INFO] Test german_data.train_gen:')
print('-'*65)
for i in range(loop):
    data = next(german_data.train_gen)
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

print('')
print('[KF INFO] Test german_data.val_gen:')
print('-'*65)
for i in range(loop):
    data = next(german_data.val_gen)
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)
