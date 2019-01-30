# 01_data_preprocessing.py
#
################################################################################
# Alibaba German AI Challenge
#  - processing the original data
#  - KF 01/30/2019 creaated
#
################################################################################
import os
import h5py 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

#===============================================================================
# Utilities
#===============================================================================
# Print title with double lines with text aligned to center
def print_title(title):
    print('')
    print('=' * 80)
    print(' ' * ((80 - len(title))//2 - 1), title)
    print('=' * 80)

# Check the folder to put plots in
plot_folder = 'plots'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

#===============================================================================
# 0. Load Data
#===============================================================================
print_title('0. Load Data')
# Data Paths
base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')
train_filename = 'training.h5'
val_filename = 'validation.h5'
round1_testA_filename = 'round1_test_a_20181109.h5'
round1_testB_filename = 'round1_test_b_20190104.h5'
round2_testA_filename = 'round2_test_a_20190121.h5'
path_training = os.path.join(base_dir, train_filename)
path_validation = os.path.join(base_dir, val_filename) 
path_round1_testA = os.path.join(base_dir, round1_testA_filename)
path_round1_testB = os.path.join(base_dir, round1_testB_filename)
path_round2_testA = os.path.join(base_dir, round2_testA_filename)

# Load Data
fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
fid_test1A = h5py.File(path_round1_testA, 'r')
fid_test1B = h5py.File(path_round1_testB, 'r')
fid_test2A = h5py.File(path_round2_testA, 'r')
#fid_test2B = h5py.File(self.path_round2_testB, 'r')

s1_training = fid_training['sen1']
s2_training = fid_training['sen2']
label_training = fid_training['label']
s1_validation = fid_validation['sen1']
s2_validation = fid_validation['sen2']
label_validation = fid_validation['label']

s1_test1A = fid_test1A['sen1']
s2_test1A = fid_test1A['sen2']
s1_test1B = fid_test1B['sen1']
s2_test1B = fid_test1B['sen2']
s1_test2A = fid_test2A['sen1']
s2_test2A = fid_test2A['sen2']
#self.s1_test2B = fid_test2B['sen1']
#self.s2_test2B = fid_test2B['sen2']

# Show data information
print("Data Info:")
print('-'*80)
print("Training data shapes:")
print('  Sentinel 1 data shape :', s1_training.shape)
print('  Sentinel 2 data shape :', s2_training.shape)
print('  Label data shape      :', label_training.shape)
print('-'*80)
print("Validation data shapes:")
print('  Sentinel 1 data shape :', s1_validation.shape)
print('  Sentinel 1 data type  :', s1_validation.dtype)
print('  Sentinel 2 data shape :', s2_validation.shape)
print('  Sentinel 2 data type  :', s2_validation.dtype)
print('  Label data shape      :', label_validation.shape)
print('  Label data type       :', label_validation.dtype)
print('-'*80)
print("Round1 TestA data shapes:")
print('  Sentinel 1 data shape :', s1_test1A.shape)
print('  Sentinel 2 data shape :', s2_test1A.shape)
print('-'*80)
print("Round1 TestB data shapes:")
print('  Sentinel 1 data shape :', s1_test1B.shape)
print('  Sentinel 2 data shape :', s2_test1B.shape)
print('-'*80)
print("Round2 TestA data shapes:")
print('  Sentinel 1 data shape :', s1_test2A.shape)
print('  Sentinel 2 data shape :', s2_test2A.shape)
print('-'*80)
print('')
print("[KF INFO] Data loaded successfully!")

#===============================================================================
# I. Original Data Distribution Analysis
#===============================================================================
print_title('I. Original Data Distribution Analysis')
#-------------------------------------------------------------------------------
# 1. Show Validation Data Distribution
#-------------------------------------------------------------------------------
# sample number sum:
val_nums = np.sum(label_validation, axis=0)
# x axis - (17 classes)
x = np.arange(17)

plt.figure(figsize=(8,6))
plt.bar(x, val_nums, width = 0.75, color='orangered')
# Annotate numbers on top or each bar.
offset = 50
focus = [0, 6, 11, 14]
others = [item for item in x if item not in focus]
for i in focus:
    xy = (i, val_nums[i]+ offset)
    plt.annotate('%d' % xy[1], xy=xy, textcoords='data', horizontalalignment='center', color='k', weight='bold')
for i in others:
    xy = (i, val_nums[i]+ offset)
    plt.annotate('%d' % xy[1], xy=xy, textcoords='data', horizontalalignment='center', color='gray')

plt.grid(axis='y', ls='--', lw=.5, c='lightgray')
plt.xticks(x)
plt.title('Original Validation Data Distribution by Labels')

plt_name = '01_original_val_data_distribution.eps'
plt.savefig(os.path.join(plot_folder, plt_name), format='eps', dpi=1000)
print('%s saved!' % plt_name)

#-------------------------------------------------------------------------------
# 2. Compare Training and Validation Data Distribution
#-------------------------------------------------------------------------------
# sample number sum:
train_nums = np.sum(label_training, axis=0)
val_nums = np.sum(label_validation, axis=0)
x = np.arange(17)

plt.figure(figsize=(24, 16))

# For Annotation
offset = 120
focus = [0, 6, 11, 14]
others = [item for item in x if item not in focus]

# Train
ax1 = plt.subplot(121)
plt.bar(x, train_nums, width = 0.75, color='dodgerblue', alpha=0.3)
for i in focus:
    xy = (i, train_nums[i]+ offset)
    ax1.annotate('%d' % xy[1], xy=xy, textcoords='data', horizontalalignment='center', color='k', weight='bold')
for i in others:
    xy = (i, train_nums[i]+ offset)
    ax1.annotate('%d' % xy[1], xy=xy, textcoords='data', horizontalalignment='center', color='gray')

plt.grid(axis='y', ls='--', lw=.5, c='lightgray')
plt.xticks(x)
plt.title('Training Data')

# Val
ax2 = plt.subplot(122, sharey=ax1)
plt.bar(x, val_nums, width = 0.75, color='orangered', alpha=0.3)
for i in focus:
    xy = (i, val_nums[i]+ offset)
    ax2.annotate('%d' % xy[1], xy=xy, textcoords='data', horizontalalignment='center', color='k', weight='bold')
for i in others:
    xy = (i, val_nums[i]+ offset)
    ax2.annotate('%d' % xy[1], xy=xy, textcoords='data', horizontalalignment='center', color='gray')

plt.grid(axis='y', ls='--', lw=.5, c='lightgray')
plt.xticks(x)
plt.title('Training Data')


plt.suptitle('Original Train vs Val Data Distribution by Labels', fontsize=24)

plt_name = '02_original_train_vs_val_data_distribution.eps'
plt.savefig(os.path.join(plot_folder, plt_name), format='eps', dpi=1000)
print('%s saved!' % plt_name)

# It turns out that class 14 has the smallest sample number:
print('-'*80)
for i in focus:
    print('Class %2d has %4d sample in total (train + val).' % (i, train_nums[i] + val_nums[i]))

#===============================================================================
# II. Data Balancing
# - Balance the dataset by adding data from training to validation set in classes
#   with insufficient samples
#===============================================================================
print_title('II. Data Balancing')
#-------------------------------------------------------------------------------
# Maximum sample numbers in validation dataset
#-------------------------------------------------------------------------------
val_max = max(val_nums)
print('Maximum sample amount in validation dataset: %4d' % val_max)
print('-'*80)
#-------------------------------------------------------------------------------
# Amount to be added in each class to validation dataset 
#-------------------------------------------------------------------------------
limit = 3000
#limit = val_max
append_count = np.zeros(17, dtype='int')
for i in np.arange(17):
    if val_nums[i] <= limit:
        min_num = min(limit, train_nums[i] + val_nums[i])
        append_count[i] = min_num - val_nums[i]
print('Amount to be added in each class to validation dataset:')
print(append_count)
print('-'*80)
#-------------------------------------------------------------------------------
# In training dataset, from END to START index, draw data samples' indices for each 
# class
#-------------------------------------------------------------------------------
# Load previous obtained indices in .npy file. 
# (if not load, the process cost about 120 seconds)
load_data = True
#load_data = False
if load_data:
    append_indices = np.load('append_indices.npy')
    print('[KF INFO] append_indices loaded!')
else:
    append_indices = []
    start = time.time()
    for cls in range(17):
        c = 0
        print('Class %d: getting indices' % cls)
        for idx in range(len(label_training))[::-1]:
            if append_count[cls] == c:
                print('%d data indices collected' % c)
                print('-'*80)
                break
            if cls == np.argmax(label_training[idx]):
                append_indices.append(idx)
                c += 1
    np.save('append_indices.npy', append_indices)
    print('[KF INFO] Processing time: %f seconds' % (time.time() - start)) # About 121 sec

print('')
print('[KF INFO] %d data indices are collected in total' % len(append_indices))


#-------------------------------------------------------------------------------
# Get (and Save) 'add' Data to h5 file:
#-------------------------------------------------------------------------------
#save_add = True
save_add = False
if save_add:
    np_s1_add, np_s2_add, np_label_add = [], [], []

    for i, idx in enumerate(append_indices):
        np_s1_add.append(s1_training[idx])
        np_s2_add.append(s2_training[idx])
        np_label_add.append(label_training[idx])
        if i > 0 and i % 500 == 0:
            print('%5d samples are loaded' % i)
            
    np_s1_add = np.array(np_s1_add)
    np_s2_add = np.array(np_s2_add)
    np_label_add = np.array(np_label_add)

    print('type  :', type(np_s1_add))
    print('length:', len(np_s1_add))
    print('[KF INFO] Saving kf_add.h5 ...')
    with h5py.File(os.path.join(base_dir, 'kf_add.h5'), 'w') as f:
        f.create_dataset('sen1', data=np_s1_add)
        f.create_dataset('sen2', data=np_s2_add)
        f.create_dataset('label', data=np_label_add)
    print('[KF INFO] kf_add.h5 saved successfully!')

else:
    print('')
    print('-'*80)
    print('[KF INFO] Loading kf_add.h5 ...')
    print('-'*80)
    fid_add = h5py.File(os.path.join(base_dir, 'kf_add.h5'), 'r')
    s1_add = fid_add['sen1']
    s2_add = fid_add['sen2']
    label_add = fid_add['label']
    print('[KF INFO] kf_add.h5 loaded!') 
    print('s1_add data shape:', s1_add.shape)
    print('s1_add data type', s1_add.dtype)
    print('s2_add data shape:', s2_add.shape)
    print('s2_add data type', s2_add.dtype)
    print('label_add data shape:', label_add.shape)
    print('label_add data type', label_add.dtype)

#-------------------------------------------------------------------------------
# Concatenate the validation and the 'add' data:
#-------------------------------------------------------------------------------
# kf_data.h5 is a copy of 'validation.h5'
#concat = True
concat = False
if concat:
    print('')
    print('-'*80)
    print('Add kf_add.h5 and validation.h5 to new file: kf_data.h5 ...')
    with h5py.File(os.path.join(base_dir, 'kf_data.h5'), 'a') as f:
        print('Creating kf_data.h5 with kf_add.h5 data ...')
        max_1 = (None, s1_add.shape[1], s1_add.shape[2], s1_add.shape[3])
        max_2 = (None, s2_add.shape[1], s2_add.shape[2], s2_add.shape[3])
        max_label = (None, label_add.shape[1])
        s1_data = f.create_dataset('sen1', data=s1_add, maxshape=max_1, chunks=True)
        s2_data = f.create_dataset('sen2', data=s2_add, maxshape=max_2, chunks=True)
        label_data = f.create_dataset('label', data=label_add, maxshape=max_label, chunks=True)
        s1_data = f['sen1']
        s2_data = f['sen2']
        label_data = f['label']
        print('Create Complete!')

        print('Concatenating validation.h5 to kf_data.h5 ...')
        val_len = label_validation.shape[0]
        s1_data.resize(s1_data.shape[0] + val_len, axis=0)
        s2_data.resize(s2_data.shape[0] + val_len, axis=0)
        label_data.resize(label_data.shape[0] + val_len, axis=0)
        s1_data[-val_len:] = s1_validation
        s2_data[-val_len:] = s2_validation
        label_data[-val_len:] = label_validation

        print('Concatenate Complete!')
        print('s1_data shape:', s1_data.shape)
        print('s2_data shape:', s2_data.shape)
        print('label_data shape:', label_data.shape)

else:
    print('')
    print('-'*80)
    print('[KF INFO] Loading kf_data.h5 ...')
    print('-'*80)
    f = h5py.File(os.path.join(base_dir, 'kf_data.h5'), 'r')
    s1_data = f['sen1']
    s2_data = f['sen2']
    label_data = f['label']

    print('[KF INFO] kf_data.h5 loaded!') 
    print('s1_data shape:', s1_data.shape)
    print('s1_data type', s1_data.dtype)
    print('s2_data shape:', s2_data.shape)
    print('s2_data type', s2_data.dtype)
    print('label_data shape:', label_data.shape)
    print('label_data type', label_data.dtype)

#-------------------------------------------------------------------------------
# Shuffle the kf_data.h5
#-------------------------------------------------------------------------------
print('')
print('-'*80)
print('Shuffle the kf_data.h5 to kf_data_shuffled.h5')
print('-'*80)
shuffle_idx = np.arange(label_data.shape[0])
np.random.seed(2019)
np.random.shuffle(shuffle_idx)
print('shuffle index shape:', shuffle_idx.shape)

with h5py.File(os.path.join(base_dir, 'kf_data_shuffled.h5'), 'w') as f:
    print('Creating kf_data_shuffled.h5')
    s1_data_shuffled = f.create_dataset('sen1', shape=s1_data.shape, dtype='float64')
    s2_data_shuffled = f.create_dataset('sen2', shape=s2_data.shape, dtype='float64')
    label_data_shuffled = f.create_dataset('label', shape=label_data.shape, dtype='float64')
    s1_data_shuffled = f['sen1']
    s2_data_shuffled = f['sen2']
    label_data_shuffled = f['label']

    print('Loop adding samples ...')
    start = time.time()
    for i in range(label_data.shape[0]):
        s1_data_shuffled[i] = s1_data[shuffle_idx[i]]
        s2_data_shuffled[i] = s2_data[shuffle_idx[i]]
        label_data_shuffled[i] = label_data[shuffle_idx[i]]
        if i % 1000 == 0 and i > 0:
            print('%5d samples processed!' % i)
            print('- time per 1000 loops: %f' % (time.time() - start))
            start = time.time()
            #print('- s1 shape', s1_data_shuffled.shape)
            #print('- dtype:', s1_data_shuffled.dtype)
            #print('- s2 shape', s2_data_shuffled.shape)
            #print('- dtype:', s2_data_shuffled.dtype)
            #print('- label  shape', label_data_shuffled.shape)
            #print('- dtype:', label_data_shuffled.dtype)
    
    print('Shuffle Complete!')



