# 01_data_preprocessing.py
#
################################################################################
# Alibaba German AI Challenge
#  - Rescale the balanced data: (e.g. kf_data_shuffled.h5, kf_val_10k.h5)
#  - KF 01/31/2019 creaated
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
plot_folder = '02_plots'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
#===============================================================================
# 0. Load Data
#===============================================================================
print_title('0. Load Data')
base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')

data_filename = 'kf_data_shuffled.h5'
val_filename = 'kf_val_10k.h5'
test_filename = 'round2_test_a_20190121.h5'

print('[KF INFO] Loading %s ...' % data_filename)
f_data = h5py.File(os.path.join(base_dir, data_filename), 'r')
s1_data = f_data['sen1']
s2_data = f_data['sen2']
label_data = f_data['label']
print('[KF INFO] Loading %s ...' % val_filename)
f_val = h5py.File(os.path.join(base_dir, val_filename), 'r')
s1_val = f_val['sen1']
s2_val = f_val['sen2']
label_val = f_val['label']
print('[KF INFO] Loading %s ...' % test_filename)
f_test = h5py.File(os.path.join(base_dir, test_filename), 'r')
s1_test = f_test['sen1']
s2_test = f_test['sen2']

print('[KF INFO] Data loaded successfully!')

#===============================================================================
# I. Bounding the Outliers
# - KF 01/31/2019
#===============================================================================
print_title('I. Bounding the Outliers')

#-------------------------------------------------------------------------------
# 1. Show 
#-------------------------------------------------------------------------------
def show_histograms_s1(dataset='data', isLog=False):
    if dataset == 'data':
        data = s1_data
    elif dataset == 'val':
        data = s1_val
    elif dataset == 'test':
        data = s1_test
        
    print('-'*80)
    print("[KF INFO] Generating histograms: dataset %s - s1 ..." % dataset)
    start = time.time()
    for ch in [4, 5, 6, 7]:
        s1 = data[..., ch]
        print('-'*80)
        print('Dataset :', dataset)
        print('Channel :', ch)
        print('Shape   :', s1.shape)
        print('Min     : ', np.min(s1))
        print('Max     : ', np.max(s1))
        print('Mean    : ', np.mean(s1))
        print('Median  : ', np.median(s1))
        print('Std     : ', np.std(s1))

        # Histograms
        #hist = np.histogram(s1, bins=100)
        #print(hist)

        print('[KF INFO] Start plotting ...')
        plt.figure(figsize=(8, 6))
        plt.hist(s1.flatten(), bins=100)
        if isLog is True:
            plt.yscale('log')
        plt.title('Histogram - %s - s1 Channel %d' % (dataset, ch))
        plt_name = '01-raw-histogram-%s-s1-ch-%d.eps' % (dataset, ch)
        plt.savefig(os.path.join(plot_folder, plt_name), format='eps', dpi=1000)
        print('[KF INFO] %s saved!' % plt_name)
        print('')
        print('[KF INFO] time spent: %s' % (time.time() - start))
        start = time.time()

show_histograms_s1('data', isLog=True)
show_histograms_s1('val', isLog=True)
show_histograms_s1('test', isLog=True)





