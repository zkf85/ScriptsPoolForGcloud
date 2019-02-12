#01_data_preprocessing.py
#
################################################################################
# Alibaba German AI Challenge
#  - Rescale the balanced data: (e.g. kf_data_shuffled.h5, kf_val_10k.h5)
#  - KF 01/31/2019 creaated
#  - KF 02/01/2019 updated
################################################################################
import os
import h5py 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import json

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

# Set base_dir as the folder including all datasets
base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')

data_filename = 'kf_data_shuffled.h5'
val_filename = 'kf_val_10k.h5'
#test2A_filename = 'round2_test_a_20190121.h5'
test2B_filename = 'round2_test_b_20190211.h5'

print('[KF INFO] Loading %s ...' % data_filename)
f_data = h5py.File(os.path.join(base_dir, data_filename), 'r')
s1_data_original = f_data['sen1']
s2_data_original = f_data['sen2']
label_data_original = f_data['label']
print('[KF INFO] Loading %s ...' % val_filename)
f_val = h5py.File(os.path.join(base_dir, val_filename), 'r')
s1_val_original = f_val['sen1']
s2_val_original = f_val['sen2']
label_val_original = f_val['label']
#print('[KF INFO] Loading %s ...' % test2A_filename)
#f_test2A = h5py.File(os.path.join(base_dir, test2A_filename), 'r')
#s1_test2A_original = f_test2A['sen1']
#s2_test2A_original = f_test2A['sen2']

print('[KF INFO] Loading %s ...' % test2B_filename)
f_test2B = h5py.File(os.path.join(base_dir, test2B_filename), 'r')
s1_test2B_original = f_test2B['sen1']
s2_test2B_original = f_test2B['sen2']
print('[KF INFO] Data loaded successfully!')

#===============================================================================
# 1. Bounding the Outliers
# - KF 01/31/2019
#===============================================================================
print_title('I. Bounding the Outliers')

#-------------------------------------------------------------------------------
# 1.1. function - plot_histograms
# - KF 02/01/2019
#-------------------------------------------------------------------------------
def plot_histograms(train_filename, val_filename, test_filename, option='original', isLog=True):
    """
    Input:
      train_filename : (str) train dataset file name (e.g. 'kf_data.h5') 
      val_filename   : (str) val dataset file (e.g. 'kf_val.h5') 
      test_filename  : (str) test dataset file (e.g. 'round2_test_a_20190121.h5')
      option         : (str) data-type (e.g. 'original', '3sigma'), description string adding to output filename
    """
    print('-'*80)
    print('[KF INFO] Start plot_histograms ...')
    print('-'*80)
    print('[KF INFO] Loading %s ...' % train_filename)
    f_data = h5py.File(os.path.join(base_dir, train_filename), 'r')
    s1_data, s2_data, label_data = f_data['sen1'], f_data['sen2'], f_data['label']
    print('[KF INFO] Loading %s ...' % val_filename)
    f_val = h5py.File(os.path.join(base_dir, val_filename), 'r')
    s1_val, s2_val, label_val = f_val['sen1'], f_val['sen2'], f_val['label']
    print('[KF INFO] Loading %s ...' % test_filename)
    f_test = h5py.File(os.path.join(base_dir, test_filename), 'r')
    s1_test, s2_test = f_test['sen1'], f_test['sen2']
    print('[KF INFO] Data loaded successfully!')
    print('-'*80)

    data = np.random.random(1000)
    #---------------------------------------------------------------------------
    # S1
    #---------------------------------------------------------------------------
    print('[KF INFO] Start plotting %s s1 histograms ...' % option)
    fig = plt.figure(figsize=(12, 8))
    for i, ch in enumerate([4, 5, 6, 7]):
        print('[KF INFO] - Concatenating s1 - channel %d ...' % ch)
        data = np.vstack((s1_data[..., ch], s1_val[..., ch], s1_test[..., ch]))
        
        print('[KF INFO] - Plotting s1 - channel %d ...' % ch)
        ax = fig.add_subplot(221 + i)
        ax.hist(data.flatten(), bins=100)
        ax.set_title('Channel: %d ' % ch)
        ax.set_yscale('log')
    plt.suptitle('Histogram - %s data S1 - (data+val+test)' % option, fontsize=24)
    plt_name = 'hist-%s-s1-data+val+test.eps' % option
    plt.savefig(os.path.join(plot_folder, plt_name), format='eps', dpi=1000)
    print('[KF INFO] %s saved!' % plt_name)
    print('-'*80)

    #---------------------------------------------------------------------------
    # S2 - only plot s2 when it's original dataset
    #---------------------------------------------------------------------------
    print('[KF INFO] Start plotting %s s2 histograms ...' % option)
    fig = plt.figure(figsize=(20, 18))
    for i, ch in enumerate(range(10)):
        print('[KF INFO] - Concatenating s2 - channel %d ...' % ch)
        data = np.vstack((s2_data[..., ch], s2_val[..., ch], s2_test[..., ch]))
        
        print('[KF INFO] - Plotting s2 - channel %d ...' % ch)
        if i == 9:
            ax = fig.add_subplot(4, 3, i + 2)
        else:
            ax = fig.add_subplot(4, 3, i + 1)
        ax.hist(data.flatten(), bins=100)
        ax.set_title('Channel: %d ' % ch)
        ax.set_yscale('log')
    plt.suptitle('Histogram - %s data S2 - (data+val+test)' % option, fontsize=32)
    plt_name = 'hist-%s-s2-data+val+test.eps' % option
    plt.savefig(os.path.join(plot_folder, plt_name), format='eps', dpi=1000)
    print('[KF INFO] %s saved!' % plt_name)
    print('-'*80)
    
# Plot original data+val+test histograms:
#tmp_data = 'kf_data_shuffled.h5'
#tmp_val = 'kf_val_10k.h5'
#tmp_test = 'round2_test_a_20190121.h5'
#plt_histograms(tmp_data, tmp_val, tmp_test)

#-------------------------------------------------------------------------------
# 1.2. Calculate the statistics of merged (data+test) datasets
#-------------------------------------------------------------------------------
def save_statistics(train_filename, test_filename):
    """
    Input:
      train_filename : (str) train dataset file name (e.g. 'kf_data.h5') 
      test_filename  : (str) test dataset file (e.g. 'round2_test_a_20190121.h5')
    """
    print('-'*80)
    print('[KF INFO] Calculate the statistics of merged (data+test) datasets ...')
    print('-'*80)
    print('[KF INFO] Loading %s ...' % train_filename)
    f_data = h5py.File(os.path.join(base_dir, train_filename), 'r')
    s1_data, s2_data, label_data = f_data['sen1'], f_data['sen2'], f_data['label']
    print('[KF INFO] Loading %s ...' % test_filename)
    f_test = h5py.File(os.path.join(base_dir, test_filename), 'r')
    s1_test, s2_test = f_test['sen1'], f_test['sen2']
    print('[KF INFO] Data loaded successfully!')
    print('-'*80)

    # Create a nested dictionary to store the statistics 
    stats = {}
    # s1:
    print('')
    print('-'*80)
    print('[KF INFO] - s1:')
    #for ch in [4,5,6,7]:
    for ch in range(8):
        print('[KF INFO] -- channel: %d ...' % ch)
        tmp_data = np.vstack((s1_data[..., ch], s1_test[..., ch]))
        tmp_min = np.min(tmp_data)
        tmp_max = np.max(tmp_data)
        tmp_mean = np.mean(tmp_data)
        tmp_median = np.median(tmp_data)
        tmp_std = np.std(tmp_data)
        stats['s1_%d' % ch] = {'min': tmp_min, 'max': tmp_max, 'mean': tmp_mean, 'median': tmp_median, 'std': tmp_std}
        print('[KF INFO] -- Done!')
    # s2
    print('')
    print('-'*80)
    print('[KF INFO] - s2:')
    for ch in range(10):
        print('[KF INFO] -- channel: %d ...' % ch)
        tmp_data = np.vstack((s2_data[..., ch], s2_test[..., ch]))
        tmp_min = np.min(tmp_data)
        tmp_max = np.max(tmp_data)
        tmp_mean = np.mean(tmp_data)
        tmp_median = np.median(tmp_data)
        tmp_std = np.std(tmp_data)
        stats['s2_%d' % ch] = {'min': tmp_min, 'max': tmp_max, 'mean': tmp_mean, 'median': tmp_median, 'std': tmp_std}
        print('[KF INFO] -- Done!')

    print('')
    print('[KF INFO] Calculation Complete!')
    stats_filename = 'statistics_full_%s+%s.json' % (train_filename.split('.')[0], test_filename.split('.')[0])
    with open(stats_filename, 'w') as f:
        json.dump(stats, f, indent=4)
    print('[KF INFO] Statistics has been saved to %s!' % stats_filename)
        
#tmp_train_filename = 'kf_data_shuffled.h5'
#tmp_test2A_filename = 'round2_test_a_20190121.h5'
#tmp_train_filename = 'kf_data_shuffled_3sigma.h5'
#tmp_test2A_filename = 'kf_test2A_3sigma.h5'

#save_statistics(tmp_train_filename, tmp_test2A_filename)

#-------------------------------------------------------------------------------
# 1.3. Bounding outliers, save to new h5 file
#-------------------------------------------------------------------------------
def bounding(train_filename, val_filename, test_filename, stats_filename, test_option='2B'):
    """
    Input:
      train_filename : (str) original train dataset file name (e.g. 'kf_data.h5') 
      val_filename   : (str) original val dataset file (e.g. 'kf_val.h5') 
      test_filename  : (str) original test dataset file (e.g. 'round2_test_a_20190121.h5')
      stats_filename : (str) statistics file of the original data 
      test_option    : (str) data-type (e.g. '2A', '2B'), string denoting '2A' or '2B', wiil be added to new file name
    """
    with open(stats_filename, 'r') as f:
        original_stats = json.load(f)

    print('[KF INFO] Loading %s ...' % train_filename)
    f_data = h5py.File(os.path.join(base_dir, train_filename), 'r')
    s1_data_original, s2_data_original, label_data_original = f_data['sen1'], f_data['sen2'], f_data['label']
    print('[KF INFO] Loading %s ...' % val_filename)
    f_val = h5py.File(os.path.join(base_dir, val_filename), 'r')
    s1_val_original, s2_val_original, label_val_original = f_val['sen1'], f_val['sen2'], f_val['label']
    print('[KF INFO] Loading %s ...' % test_filename)
    f_test = h5py.File(os.path.join(base_dir, test_filename), 'r')
    s1_test_original, s2_test_original = f_test['sen1'], f_test['sen2']
    print('[KF INFO] Data loaded successfully!')
    print('-'*80)

    for factor in [3]:
        # Set bounding factor (+- factor*std is the boundary)
        new_data_filename = '%s_%dsigma.h5' % (data_filename.split('.')[0], factor)
        new_val_filename = '%s_%dsigma.h5' % (val_filename.split('.')[0], factor)
        new_test_filename = 'kf_test%s_%dsigma.h5' % (test_option, factor)
        
        # 1. new kf_data
        #with h5py.File(os.path.join(base_dir, new_data_filename), 'w') as f:
        #    print('-'*80)
        #    print('[KF INFO] Creating %s ...' % new_data_filename)
        #    s1 = f.create_dataset('sen1', shape=s1_data_original.shape, dtype='float64')
        #    s2 = f.create_dataset('sen2', shape=s2_data_original.shape, dtype='float64')
        #    label = f.create_dataset('label', shape=label_data_original.shape, dtype='float64')
        #    print('s1 shape   :', s1.shape)
        #    print('s2 shape   :', s2.shape)
        #    print('label shape:', label.shape)
        #    # s1
        #    print('[KF INFO] - s1 ...')
        #    #for ch in [4,5,6,7]:
        #    for ch in range(8):
        #        print('[KF INFO] -- channel %d' % ch)
        #        tmp_data = s1_data_original[..., ch]
        #        tmp_med = original_stats['s1_%d' % ch]['median']
        #        tmp_std = original_stats['s1_%d' % ch]['std']
        #        print('[KF INFO] --- Start Bounding ...')
        #        start = time.time()
        #        np.clip(tmp_data, tmp_med - factor * tmp_std, tmp_med + factor * tmp_std, out=tmp_data)
        #        s1[..., ch] = tmp_data
        #        print('[KF INFO] --- done in %s seconds!' % (time.time() - start))
        #        start = time.time()
        #    # Save s2 and labels without any change
        #    print('-'*80)
        #    print('[KF INFO] - s2 and label ...')
        #    for i in range(label_data_original.shape[0]):
        #        s2[i] = s2_data_original[i]
        #        label[i] = label_data_original[i]
        #        if i > 0 and i % 1000 == 0:
        #            print('[KF INFO] -- %5d rows are processed!' % i)
        #print('[KF INFO] %s is saved!' % new_data_filename)
        #print('')

        ## 2. new kf_val
        #with h5py.File(os.path.join(base_dir, new_val_filename), 'w') as f:
        #    print('-'*80)
        #    print('[KF INFO] Creating %s ...' % new_val_filename)
        #    s1 = f.create_dataset('sen1', shape=s1_val_original.shape, dtype='float64')
        #    s2 = f.create_dataset('sen2', shape=s2_val_original.shape, dtype='float64')
        #    label = f.create_dataset('label', shape=label_val_original.shape, dtype='float64')
        #    print('s1 shape   :', s1.shape)
        #    print('s2 shape   :', s2.shape)
        #    print('label shape:', label.shape)
        #    # s1
        #    print('[KF INFO] - s1 ...')
        #    for ch in [4,5,6,7]:
        #        print('[KF INFO] -- channel %d' % ch)
        #        tmp_data = s1_val_original[..., ch]
        #        tmp_med = original_stats['s1_%d' % ch]['median']
        #        tmp_std = original_stats['s1_%d' % ch]['std']
        #        print('[KF INFO] --- Start Bounding ...')
        #        start = time.time()
        #        np.clip(tmp_data, tmp_med - factor * tmp_std, tmp_med + factor * tmp_std, out=tmp_data)
        #        s1[..., ch] = tmp_data
        #        print('[KF INFO] --- done in %s seconds!' % (time.time() - start))
        #        start = time.time()
        #    # Save s2 and labels without any change
        #    print('-'*80)
        #    print('[KF INFO] - s2 and label ...')
        #    for i in range(label_val_original.shape[0]):
        #        s2[i] = s2_val_original[i]
        #        label[i] = label_val_original[i]
        #        if i > 0 and i % 1000 == 0:
        #            print('[KF INFO] -- %5d rows are processed!' % i)
        #print('[KF INFO] %s is saved!' % new_data_filename)
        #print('')

        # 3. new kf_test
        with h5py.File(os.path.join(base_dir, new_test_filename), 'w') as f:
            print('-'*80)
            print('[KF INFO] Creating %s ...' % new_test_filename)
            s1 = f.create_dataset('sen1', shape=s1_test_original.shape, dtype='float64')
            s2 = f.create_dataset('sen2', shape=s2_test_original.shape, dtype='float64')
            print('s1 shape   :', s1.shape)
            print('s2 shape   :', s2.shape)
            # s1
            print('[KF INFO] - s1 ...')
            for ch in [4,5,6,7]:
                print('[KF INFO] -- channel %d' % ch)
                tmp_data = s1_test_original[..., ch]
                tmp_med = original_stats['s1_%d' % ch]['median']
                tmp_std = original_stats['s1_%d' % ch]['std']
                print('[KF INFO] --- Start Bounding ...')
                start = time.time()
                np.clip(tmp_data, tmp_med - factor * tmp_std, tmp_med + factor * tmp_std, out=tmp_data)
                s1[..., ch] = tmp_data
                print('[KF INFO] --- done in %s seconds!' % (time.time() - start))
                start = time.time()
            # Save s2 without any change 
            print('-'*80)
            print('[KF INFO] - s2 ...')
            for i in range(s2_test_original.shape[0]):
                s2[i] = s2_test_original[i]
                if i > 0 and i % 1000 == 0:
                    print('[KF INFO] -- %5d rows are processed!' % i)

        print('[KF INFO] %s is saved!' % new_test_filename)
        print('')

        # 4. Plot histograms for new data with (factor * sigma)
        #plot_histograms(new_data_filename, new_val_filename, new_test_filename, option='%dsigma' % factor)

raw_stats = 'statistics_original_kf_data+test_2A.json'
raw_data = 'kf_data_shuffled.h5'
raw_val = 'kf_val_10k.h5'
#raw_test = 'round2_test_a_20190121.h5'
raw_test = 'round2_test_b_20190211.h5'
bounding(raw_data, raw_val, raw_test, raw_stats)

#===============================================================================
# 2. Rescaling (standardize / max-min rescale)
# - KF 02/02/2019
#===============================================================================
def rescale(train_filename, val_filename, test_filename, stats_filename, method='standardize'):
    """
    Input:
      train_filename : (str) original train dataset file name (e.g. 'kf_data.h5') 
      val_filename   : (str) original val dataset file (e.g. 'kf_val.h5') 
      test_filename  : (str) original test dataset file (e.g. 'round2_test_a_20190121.h5')
      stats_filename : (str) statistics file of the original data 
      method         : (str) either 'standardize' or 'max-min'
    """
    print_title('II. Rescaling')
    # Load stats: 
    with open(stats_filename, 'r') as f:
        stats = json.load(f)
        print('[KF INFO] Stats loaded from %s' % stats_filename)
    # Load datasets   
    print('[KF INFO] Loading %s ...' % train_filename)
    f_data = h5py.File(os.path.join(base_dir, train_filename), 'r')
    s1_data, s2_data, label_data = f_data['sen1'], f_data['sen2'], f_data['label']
    print('[KF INFO] Loading %s ...' % val_filename)
    f_val = h5py.File(os.path.join(base_dir, val_filename), 'r')
    s1_val, s2_val, label_val = f_val['sen1'], f_val['sen2'], f_val['label']
    print('[KF INFO] Loading %s ...' % test_filename)
    f_test = h5py.File(os.path.join(base_dir, test_filename), 'r')
    s1_test, s2_test = f_test['sen1'], f_test['sen2']
    print('[KF INFO] Data loaded successfully!')
    print('-'*80)

    # Set new file names
    if method == 'standardize':
        name_suffix = 'standardized'
    elif method == 'max-min':
        name_suffix = 'max-min-rescaled' 

    new_data_filename = '%s_%s.h5' % (train_filename.split('.')[0], name_suffix)
    new_val_filename = '%s_%s.h5' % (val_filename.split('.')[0], name_suffix)
    new_test_filename = '%s_%s.h5' % (test_filename.split('.')[0], name_suffix)
    # Get the bounding facter string (e.g. '1', '2', '3')
    factor_str = train_filename.split('sigma')[0][-1]

    ## 1. new train(data) file
    #with h5py.File(os.path.join(base_dir, new_data_filename), 'w') as f:
    #    print('-'*80)
    #    print('[KF INFO] Creating %s ...' % new_data_filename)
    #    s1 = f.create_dataset('sen1', shape=s1_data.shape, dtype='float64')
    #    s2 = f.create_dataset('sen2', shape=s2_data.shape, dtype='float64')
    #    label = f.create_dataset('label', shape=label_data.shape, dtype='float64')
    #    # s1
    #    print('[KF INFO] - s1 ...')
    #    for ch in [4,5,6,7]:
    #        print('[KF INFO] -- channel %d' % ch)
    #        tmp_data = s1_data[..., ch]
    #        tmp_mean = stats['s1_%d' % ch]['mean']
    #        tmp_std = stats['s1_%d' % ch]['std']
    #        tmp_min = stats['s1_%d' % ch]['min']
    #        tmp_max = stats['s1_%d' % ch]['max']
    #        start = time.time()
    #        if method == 'standardize':
    #            print('[KF INFO] --- Start standardize ...')
    #            s1[..., ch] = (tmp_data - tmp_mean) / tmp_std
    #        elif method == 'max-min':
    #            print('[KF INFO] --- Start Max-Min rescaling ...')
    #            s1[..., ch] = (tmp_data - tmp_min) / (tmp_max - tmp_min)
    #        else:
    #            raise Exception('[KF ERROR] No method selected!')
    #        print('[KF INFO] --- done in %s seconds!' % (time.time() - start))
    #        start = time.time()
    #    # s2
    #    print('[KF INFO] - s2 ...')
    #    for ch in range(10):
    #        print('[KF INFO] -- channel %d' % ch)
    #        tmp_data = s2_data[..., ch]
    #        tmp_mean = stats['s2_%d' % ch]['mean']
    #        tmp_std = stats['s2_%d' % ch]['std']
    #        tmp_min = stats['s2_%d' % ch]['min']
    #        tmp_max = stats['s2_%d' % ch]['max']
    #        start = time.time()
    #        if method == 'standardize':
    #            print('[KF INFO] --- Start standardize ...')
    #            s2[..., ch] = (tmp_data - tmp_mean) / tmp_std
    #        elif method == 'max-min':
    #            print('[KF INFO] --- Start Max-Min rescaling ...')
    #            s2[..., ch] = (tmp_data - tmp_min) / (tmp_max - tmp_min)
    #        else:
    #            raise Exception('[KF ERROR] No method selected!')
    #        print('[KF INFO] --- done in %s seconds!' % (time.time() - start))
    #        start = time.time()
    #    # Save labels without any change
    #    print('-'*80)
    #    print('[KF INFO] - label ...')
    #    for i in range(label_data.shape[0]):
    #        label[i] = label_data[i]
    #        if i > 0 and i % 1000 == 0:
    #            print('[KF INFO] -- %5d rows are processed!' % i)
    #print('[KF INFO] %s is saved!' % new_data_filename)
    #print('')

    ## 2. new val file
    #with h5py.File(os.path.join(base_dir, new_val_filename), 'w') as f:
    #    print('-'*80)
    #    print('[KF INFO] Creating %s ...' % new_val_filename)
    #    s1 = f.create_dataset('sen1', shape=s1_val.shape, dtype='float64')
    #    s2 = f.create_dataset('sen2', shape=s2_val.shape, dtype='float64')
    #    label = f.create_dataset('label', shape=label_val.shape, dtype='float64')
    #    # s1
    #    print('[KF INFO] - s1 ...')
    #    for ch in [4,5,6,7]:
    #        print('[KF INFO] -- channel %d' % ch)
    #        tmp_data = s1_val[..., ch]
    #        tmp_mean = stats['s1_%d' % ch]['mean']
    #        tmp_std = stats['s1_%d' % ch]['std']
    #        tmp_min = stats['s1_%d' % ch]['min']
    #        tmp_max = stats['s1_%d' % ch]['max']
    #        start = time.time()
    #        if method == 'standardize':
    #            print('[KF INFO] --- Start standardize ...')
    #            s1[..., ch] = (tmp_data - tmp_mean) / tmp_std
    #        elif method == 'max-min':
    #            print('[KF INFO] --- Start Max-Min rescaling ...')
    #            s1[..., ch] = (tmp_data - tmp_min) / (tmp_max - tmp_min)
    #        else:
    #            raise Exception('[KF ERROR] No method selected!')
    #        print('[KF INFO] --- done in %s seconds!' % (time.time() - start))
    #        start = time.time()
    #    # s2
    #    print('[KF INFO] - s2 ...')
    #    for ch in range(10):
    #        print('[KF INFO] -- channel %d' % ch)
    #        tmp_data = s2_val[..., ch]
    #        tmp_mean = stats['s2_%d' % ch]['mean']
    #        tmp_std = stats['s2_%d' % ch]['std']
    #        tmp_min = stats['s2_%d' % ch]['min']
    #        tmp_max = stats['s2_%d' % ch]['max']
    #        start = time.time()
    #        if method == 'standardize':
    #            print('[KF INFO] --- Start standardize ...')
    #            s2[..., ch] = (tmp_data - tmp_mean) / tmp_std
    #        elif method == 'max-min':
    #            print('[KF INFO] --- Start Max-Min rescaling ...')
    #            s2[..., ch] = (tmp_data - tmp_min) / (tmp_max - tmp_min)
    #        else:
    #            raise Exception('[KF ERROR] No method selected!')
    #        print('[KF INFO] --- done in %s seconds!' % (time.time() - start))
    #        start = time.time()
    #    # Save labels without any change
    #    print('-'*80)
    #    print('[KF INFO] - label ...')
    #    for i in range(label_val.shape[0]):
    #        label[i] = label_val[i]
    #        if i > 0 and i % 1000 == 0:
    #            print('[KF INFO] -- %5d rows are processed!' % i)
    #print('[KF INFO] %s is saved!' % new_val_filename)
    #print('')

    # 3. new test file
    with h5py.File(os.path.join(base_dir, new_test_filename), 'w') as f:
        print('-'*80)
        print('[KF INFO] Creating %s ...' % new_test_filename)
        s1 = f.create_dataset('sen1', shape=s1_test.shape, dtype='float64')
        s2 = f.create_dataset('sen2', shape=s2_test.shape, dtype='float64')
        # s1
        print('[KF INFO] - s1 ...')
        for ch in [4,5,6,7]:
            print('[KF INFO] -- channel %d' % ch)
            tmp_data = s1_test[..., ch]
            tmp_mean = stats['s1_%d' % ch]['mean']
            tmp_std = stats['s1_%d' % ch]['std']
            tmp_min = stats['s1_%d' % ch]['min']
            tmp_max = stats['s1_%d' % ch]['max']
            start = time.time()
            if method == 'standardize':
                print('[KF INFO] --- Start standardize ...')
                s1[..., ch] = (tmp_data - tmp_mean) / tmp_std
            elif method == 'max-min':
                print('[KF INFO] --- Start Max-Min rescaling ...')
                s1[..., ch] = (tmp_data - tmp_min) / (tmp_max - tmp_min)
            else:
                raise Exception('[KF ERROR] No method selected!')
            print('[KF INFO] --- done in %s seconds!' % (time.time() - start))
            start = time.time()
        # s2
        print('[KF INFO] - s2 ...')
        for ch in range(10):
            print('[KF INFO] -- channel %d' % ch)
            tmp_data = s2_test[..., ch]
            tmp_mean = stats['s2_%d' % ch]['mean']
            tmp_std = stats['s2_%d' % ch]['std']
            tmp_min = stats['s2_%d' % ch]['min']
            tmp_max = stats['s2_%d' % ch]['max']
            start = time.time()
            if method == 'standardize':
                print('[KF INFO] --- Start standardize ...')
                s2[..., ch] = (tmp_data - tmp_mean) / tmp_std
            elif method == 'max-min':
                print('[KF INFO] --- Start Max-Min rescaling ...')
                s2[..., ch] = (tmp_data - tmp_min) / (tmp_max - tmp_min)
            else:
                raise Exception('[KF ERROR] No method selected!')
            print('[KF INFO] --- done in %s seconds!' % (time.time() - start))
            start = time.time()
    print('[KF INFO] %s is saved!' % new_test_filename)
    print('')

    # 4. Plot histograms for new data with (factor * sigma)
    #plot_histograms(new_data_filename, new_val_filename, new_test_filename, option='%ssigma_%s' % (factor_str, name_suffix))

    # 5. Save statistics:
    save_statistics(new_data_filename, new_test_filename)

tmp_train = 'kf_data_shuffled_3sigma.h5'
tmp_val = 'kf_val_10k_3sigma.h5'
#tmp_test = 'kf_test2A_3sigma.h5'
tmp_test = 'kf_test2B_3sigma.h5'
tmp_stats = 'statistics_kf_data_shuffled_3sigma+kf_test2A_3sigma.json'
rescale(tmp_train, tmp_val, tmp_test, tmp_stats, method='standardize')
#rescale(tmp_train, tmp_val, tmp_test, tmp_stats, method='max-min')



