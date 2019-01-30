# KF 01/28/2019
################################################################################
# Alibaba Cloud German AI Challenge 2018
################################################################################
# I. Imports and Paths
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import time

# Print title with double lines with text aligned to center
def print_title(title, width=80):
    print('')
    print('=' * width)
    print(' ' * ((width - len(title))//2 - 1), title)
    print('=' * width)

#===============================================================================
# I. Validate the paths
#===============================================================================
# Set paths
base_dir = os.path.expanduser('/home/kefeng/German_AI_Challenge/dataset')
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')

# Validate the paths
print_title('I. Validate the Data Files')
for f in os.listdir(base_dir):
    print(f)

#===============================================================================
# II. Overview of the Data
#===============================================================================
fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')

# Have a look at keys stored in the h5 files
print_title("Overview of Data:")
print('Training data keys   :', list(fid_training.keys()))
print('Validation data keys :', list(fid_validation.keys()))

print('-'*80)
print("Training data shapes:")
s1_training = fid_training['sen1']
s2_training = fid_training['sen2']
label_training = fid_training['label']
print('  Sentinel 1 data shape :', s1_training.shape)
print('  Sentinel 2 data shape :', s2_training.shape)
print('  Label data shape      :', label_training.shape)
print('-'*80)
print("Validation data shapes:")
s1_validation = fid_validation['sen1']
s2_validation = fid_validation['sen2']
label_validation = fid_validation['label']
print('  Sentinel 1 data shape :', s1_validation.shape)
print('  Sentinel 2 data shape :', s2_validation.shape)
print('  Label data shape      :', label_validation.shape)

#===============================================================================
# Statistics 
#===============================================================================
# S1, Channel 4, 5, 6, 7
#val_s1 = s1_validation[...,-1]
def show_histograms_s1():
    print("s1 validation histograms:")
    for ch in [4, 5, 6, 7]:

        val_s1 = s1_validation[..., ch]
        print('Channel : ', ch)
        print('Shape:', val_s1.shape)
        print('Min  : ', np.min(val_s1))
        print('Max  : ', np.max(val_s1))
        print('Mean : ', np.mean(val_s1))
        print('Std  : ', np.std(val_s1))

        # Histograms
        hist = np.histogram(val_s1, bins=100)
        print(hist)

        plt.figure(figsize=(8, 6))
        plt.hist(val_s1.flatten(), bins=100)
        if ylog is True:
            plt.yscale('log')
        plt.title('Histogram - s1 Channel %d' % ch)
        plt_name = '01-raw-histogram-s1-ch-%d.eps' % ch
        plt.savefig(plt_name, format='eps', dpi=1000)

    # S2, Channel All
def show_histograms_s2():
    print("s2 validation histograms:")
    for ch in range(s2_validation.shape[-1]):

        val_s2 = s2_validation[..., ch]
        print('Channel : ', ch)
        print('Shape:', val_s2.shape)
        print('Min  : ', np.min(val_s2))
        print('Max  : ', np.max(val_s2))
        print('Mean : ', np.mean(val_s2))
        print('Std  : ', np.std(val_s2))

        # Histograms
        hist = np.histogram(val_s2, bins=100)
        print(hist)
        fig = plt.figure(figsize=(6, 8))

        ax = fig.add_subplot(2, 1, 1)
        ax.hist(val_s2.flatten(), bins=100)
        
        ax = fig.add_subplot(2, 1, 2)
        ax.hist(val_s2.flatten(), bins=100)
        ax.set_yscale('log')

        fig.suptitle('Histogram - s2 Channel %d' % ch)
        fig.subplots_adjust(hspace = 0.3)
        plt_name = '01-raw-histogram-s2-ch-%d.eps' % ch
        plt.savefig(plt_name, format='eps', dpi=1000)

#show_histograms_s1()
#show_histograms_s2()

#===============================================================================
# Remove/Replace Outliers
#===============================================================================
#def reject_outliers(data, m=1):
#    return data[abs(data-np.mean(data)) < m * np.std(data)]
#def replace_outliers(data, med, std, m=3):
#    if abs(data - med) > m * std:
#        print(data)
#        data = med

def replace_outliers(ylog=True, normalize=False):
    print_title('Replace Outliers with Median')

    fig = plt.figure(figsize=(16, 16))
    fig_n = 1
    
    for ch in [4, 5, 6, 7]:
        ax = fig.add_subplot(5, 4, fig_n)
        data = s1_validation[..., ch]
        ax.hist(data.flatten(), bins=100)
        ax.set_title('Channel: %d  - Original' % ch)
        if ylog is True:
            ax.set_yscale('log')
        fig_n += 1

    loop = 0
    for m in [1.0, 1.5, 2.0, 3.0]:
        for ch in [4, 5, 6, 7]:
            loop += 1
            print('-'*80)
            print("Main Loop Number %d" % loop)
            print('-'*80)

            counter = 0
            data = s1_validation[..., ch]
            med = np.median(data)
            std = np.std(data)

            print('Loop starts ...')
            start = time.time()
            for (n, w, h), value in np.ndenumerate(data):
                if data[n, w, h] - med > m * std:
                    counter += 1
                    data[n, w, h] = m * std
                elif data[n, w, h] - med < - m * std:
                    counter += 1
                    data[n, w, h] = - m * std

                #if abs(data[n, w, h]) > th:
                #    counter += 1
                #    print('data in [%d] larger than %f: %f' % (n, data[n, w, h], th))
                #    print(data[n])
                #    data[n, w, h] = th

            # Standardize
            #print('Standardize ...')
            #data = data / np.max(data)
            
            # Normalization
            if normalize is True:
                print('Normalizing ...')
                for i in range(len(data)):
                    norm = np.linalg.norm(data[i])
                    if i % 100 == 0:
                        print(norm)
                    data[i] /= norm

            print('Loop time : ', time.time() - start)
            print('Replaced %d outliers' % counter)

            print('Channel : ', ch)
            print('Shape  :', data.shape)
            print('Min    : ', np.min(data))
            print('Max    : ', np.max(data))
            print('Mean   : ', np.mean(data))
            print('Median : ', np.median(data))
            print('Std    : ', np.std(data))
            # Histograms
            #hist = np.histogram(data, bins=100)
            #print(hist)
            ax = fig.add_subplot(5, 4, fig_n)
            ax.hist(data.flatten(), bins=100)
            ax.set_title('Channel: %d  - Threshold: %1.1f' % (ch, m))
            if ylog is True:
                ax.set_yscale('log')

            fig_n += 1

    fig.subplots_adjust(hspace = 0.3)

    suptitle = 'Histograms S1 Remove Outlier - Channel/Threshold'
    plt_name = '02-histogram-remove-outlier-s1'
    if normalize is True:
        suptitle += ' -Normalized'
        plt_name += '-normalized'
    if ylog is True:
        suptitle += ' (log)'
        plt_name += '-log'    

    fig.suptitle(suptitle, fontsize=20)
    plt.savefig(plt_name+'.eps', format='eps', dpi=1000)

    
#replace_outliers()
#replace_outliers(ylog=False)
replace_outliers(ylog=True, normalize=True)

#===============================================================================
# Data Normalization
#===============================================================================
def normalize():
    print_title('Data Normalization')
    ch = 5
    data = s1_validation[..., ch]

    counter = 0
    start = time.time()
    print('Loop start ...')
    for i in range(len(data)):
        counter += 1
        norm = np.linalg.norm(data[i])
        if norm > 100:
            print('norm [%d] larger than 100: %f' % (i, norm))
            print(data[i])
        data[i] /= norm

    print('Loop number:', counter)
    print('Loop time:', time.time() - start)

    print('Channel : ', ch)
    print('Shape  :', data.shape)
    print('Min    : ', np.min(data))
    print('Max    : ', np.max(data))
    print('Mean   : ', np.mean(data))
    print('Median : ', np.median(data))
    print('Std    : ', np.std(data))

    ## Histograms
    hist = np.histogram(data, bins=100)
    print(hist)

    plt.figure(figsize=(8, 6))
    plt.hist(data.flatten(), bins=100)
    plt.title('Histogram (normalized) - s1 Channel %d' % ch)
    plt_name = 'histogram-normalized-try.eps'
    plt.savefig(plt_name, format='eps', dpi=1000)

#normalize()


