# 01_dummy.py
#################################################################
# Alibaba Cloud German AI Challenge 2018
# KF 2018/12/17
#
# Make a Balanced (but smaller) Dataset
#################################################################
import os
import numpy as np
import h5py
import random

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

# Have a look at keys stored in the h5 files
print_title("Overview of Data:")
print('Training data keys   :', list(fid_training.keys()))
print('Validation data keys :', list(fid_validation.keys()))

print('-'*65)
print("Training data shapes:")
s1_training = fid_training['sen1']
s2_training = fid_training['sen2']
label_training = fid_training['label']
print('  Sentinel 1 data shape :', s1_training.shape)
print('  Sentinel 2 data shape :', s2_training.shape)
print('type:', s1_training.dtype)
print('  Label data shape      :', label_training.shape)
print('type:', label_training.dtype)
print('-'*65)
print("Validation data shapes:")
s1_validation = fid_validation['sen1']
s2_validation = fid_validation['sen2']
label_validation = fid_validation['label']
print('  Sentinel 1 data shape :', s1_validation.shape)
print('  Sentinel 2 data shape :', s2_validation.shape)
print('  Label data shape      :', label_validation.shape)

# KF 12/17/2018
# Sample Balancing
label_all = np.concatenate((label_training, label_validation), axis=0)
label_qty = np.sum(label_all, axis=0)
min_size = int(np.min(label_qty))
print("Minimal class sample size : ", min_size)

# convert one hot to explicit category
label_all_cat = np.array(np.argmax(label_all, axis=1))
print("training length  :", len(label_training))
print("validation length:", len(label_validation))
print("Label_all length :", len(label_all_cat))

# Build indices list for each category
cls_list = [[] for i in range(17)]
for idx, cls in enumerate(label_all_cat):
    cls_list[cls].append(idx)
print(len(cls_list))

small_cls_list = []
for i, l in enumerate(cls_list):
    small_cls_list += random.sample(l, min_size)
    print(len(small_cls_list))
    
print(len(small_cls_list))

## CAUTION: This block requires 16G Memory!!!
#s1, s2, label = [], [], []
#for li in small_cls_list:
#    for i in li:
#        
#        if i >= len(label_training):
#            s1.append(s1_validation[i - len(label_training)])
#            s2.append(s2_validation[i - len(label_training)])
#            label.append(label_validation[i - len(label_training)])
#        else:
#            s1.append(s1_training[i])
#            s2.append(s2_training[i])
#            label.append(label_training[i])
#
#    print(len(s1), 'Done!')
#
#with h5py.File('small_balanced_dataset.h5', 'w') as f:
#    f['s1'] = np.array(s1)
#    f['s2'] = np.array(s2)
#    f['label'] = np.array(label)



