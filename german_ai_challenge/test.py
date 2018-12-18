# test.py
#################################################################
# Created by 
#   KF 12/18/2018
#
# Re-construct the code for Ali-German AI Challenger AI Competition
# Distribute functions into modules.
#
#################################################################
import os

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

#################################################################
# Test Initialization 
#################################################################
from kfdata.KFGermanData import GermanData

german_data = GermanData(param_dict)

#----------------------------------------------------------------
# Test get data shape:
#----------------------------------------------------------------
print('')
print("[KF INFO] Test getDataShape with channel 'full':")
print("  data shape:", german_data.getDataShape(channel='full'))

print('')
print("[KF INFO] Test getDataShape with channel 's2_rgb':")
print("  data shape:", german_data.getDataShape(channel='s2_rgb'))

#################################################################
# Test Generators:
#################################################################
batch_size = 64
loop = 10
print('')
print('[KF INFO] Test trainGenerator:')
print('-'*65)
for i in range(loop):
    data = next(german_data.trainGenerator(batch_size, train_mode='real'))
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

print('')
print('[KF INFO] Test valGenerator:')
print('-'*65)
for i in range(loop):
    data = next(german_data.valGenerator(batch_size, train_mode='real'))
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

print('')
print('[KF INFO] Test balancedTrainGenerator:')
print('-'*65)
for i in range(loop):
    data = next(german_data.balancedTrainGenerator(batch_size, train_mode='real'))
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

print('')
print('[KF INFO] Test balancedValGenerator:')
print('-'*65)
for i in range(loop):
    data = next(german_data.balancedValGenerator(batch_size, train_mode='real'))
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

print('')
print('[KF INFO] Test S2 RGB channel for trainGenerator:')
print('-'*65)
for i in range(loop):
    data = next(german_data.trainGenerator(batch_size, channel='s2_rgb', train_mode='test'))
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

print('')
print('[KF INFO] Test S2 RGB channel for valGenerator:')
print('-'*65)
for i in range(loop):
    data = next(german_data.valGenerator(batch_size, channel='s2_rgb', train_mode='test'))
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

print('')
print('[KF INFO] Test S2 RGB channel for balancedTrainGenerator:')
print('-'*65)
for i in range(loop):
    data = next(german_data.balancedTrainGenerator(batch_size, channel='s2_rgb', train_mode='test'))
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

print('')
print('[KF INFO] Test S2 RGB channel for balancedValGenerator:')
print('-'*65)
for i in range(loop):
    data = next(german_data.balancedValGenerator(batch_size, channel='s2_rgb', train_mode='test'))
    print("batch_X shape:", data[0].shape, "batch_y shape:", data[1].shape)

