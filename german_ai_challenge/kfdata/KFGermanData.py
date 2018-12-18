#KFGermanData.py
# KF 12/17/2018

import os
import h5py
import numpy as np
import random

class GermanData:

    #####################################################################
    #                          Initialization
    #####################################################################
    def __init__(self, params):
        # Initialize paths parameters
        self.base_dir = params.get('base_dir')

        self.train_filename = params.get('train_filename')
        self.val_filename = params.get('val_filename')
        self.round1_test_filename = params.get('round1_test_filename')

        self.path_training = os.path.join(self.base_dir, self.train_filename)
        self.path_validation = os.path.join(self.base_dir, self.val_filename) 
        self.path_round1_test = os.path.join(self.base_dir, self.round1_test_filename)

        #----------------------------------------------------------------
        # 1. Paths Validation
        #----------------------------------------------------------------
        self.print_title('Load Data')
        print("[KF INFO] Validate data file existance: ")
        for f in [self.train_filename, self.val_filename, self.round1_test_filename]:
            if f in os.listdir(self.base_dir):
                print('|--CHECK!-->', f)
        print('')

        #----------------------------------------------------------------
        # 2. Load Data
        #----------------------------------------------------------------
        fid_training = h5py.File(self.path_training, 'r')
        fid_validation = h5py.File(self.path_validation, 'r')
        fid_test1 = h5py.File(self.path_round1_test, 'r')

        self.s1_training = fid_training['sen1']
        self.s2_training = fid_training['sen2']
        self.label_training = fid_training['label']
        self.s1_validation = fid_validation['sen1']
        self.s2_validation = fid_validation['sen2']
        self.label_validation = fid_validation['label']
        self.s1_test1 = fid_test1['sen1']
        self.s2_test1 = fid_test1['sen2']
        print("[KF INFO] Data loaded successfully!")

        self.train_size = len(self.label_training)
        self.val_size = len(self.label_validation)

        #----------------------------------------------------------------
        # Set test parameters 
        #----------------------------------------------------------------
        self.small_train_size = 4000
        self.small_val_size = 1000

        self.showDataInfo()
        self.getBalancedData()


    #===================================================================
    # Print title with double lines with text aligned to center
    #===================================================================
    @staticmethod
    def print_title(title):
        print('')
        print('=' * 65)
        print(' ' * ((65 - len(title))//2 - 1), title)
        print('=' * 65)
   

    #===================================================================
    # Get Data Info :
    #===================================================================
    def showDataInfo(self):

        self.print_title("Data Info:")
        # Show keys stored in the h5 files
        #print('Training data keys   :', list(fid_training.keys()))
        #print('Validation data keys :', list(fid_validation.keys()))
        #print('Round1 Test data keys :', list(fid_test1.keys()))
        #print('-'*65)
        print("Training data shapes:")
        print('  Sentinel 1 data shape :', self.s1_training.shape)
        print('  Sentinel 2 data shape :', self.s2_training.shape)
        print('  Label data shape      :', self.label_training.shape)
        print('-'*65)
        print("Validation data shapes:")
        print('  Sentinel 1 data shape :', self.s1_validation.shape)
        print('  Sentinel 2 data shape :', self.s2_validation.shape)
        print('  Label data shape      :', self.label_validation.shape)
        print('-'*65)
        print("Round1 Test data shapes:")
        print('  Sentinel 1 data shape :', self.s1_test1.shape)
        print('  Sentinel 2 data shape :', self.s2_test1.shape)


    #===================================================================
    # Get balance data :
    #   Save balanced data as class variables
    #===================================================================
    def getBalancedData(self):
        self.print_title('Get Balanced Data')
        label_all = np.concatenate((self.label_training, self.label_validation), axis=0)
        label_qty = np.sum(label_all, axis=0)
        min_size = int(np.min(label_qty))
        print("Minimal class sample size : ", min_size)

        # convert one hot to explicit category
        label_all_cat = np.array(np.argmax(label_all, axis=1))

        # Build indices list for each category
        cls_list = [[] for i in range(17)]
        for idx, cls in enumerate(label_all_cat):
            cls_list[cls].append(idx)

        balanced_idx_list = []
        for i, l in enumerate(cls_list):
            balanced_idx_list += random.sample(l, min_size)

        # This variable contains a balanced version of samples for later training
        sorted_balanced_idx_list = sorted(balanced_idx_list)
        #print(sorted_balanced_idx_list[:20])
        print("Balanced dataset size     :", len(sorted_balanced_idx_list))

        # Validate the distribution of the balanced dataset:
        print('')
        print('-'*65)
        print("[KF INFO] Validate the balanced dataset:")
        test_res = np.zeros((17,))
        for idx in sorted_balanced_idx_list:
            if idx >= self.train_size:
                label_tmp = self.label_validation[idx - self.train_size]
            else:
                label_tmp = self.label_training[idx]
            test_res += label_tmp
        for i, num in enumerate(test_res):
            print("  number of samples in class %2d :" % i, int(num))

        # Shuffle inplace:
        random.shuffle(sorted_balanced_idx_list)
        # Divide training and validation dataset
        split = len(sorted_balanced_idx_list) *4 // 5

        # Save to variables
        self.balanced_train_idx_list = sorted_balanced_idx_list[:split]
        self.balanced_val_idx_list = sorted_balanced_idx_list[split:]
        self.balanced_train_size = len(self.balanced_train_idx_list)
        self.balanced_val_size = len(self.balanced_val_idx_list)


    #####################################################################
    #                         Data Generators
    #####################################################################

    #===================================================================
    # Get data shape
    #===================================================================
    def getDataShape(self, channel='full'):

        input_width = self.s1_training.shape[1] 
        input_height = self.s1_training.shape[2]

        if channel == 'full':
            # channel number should be 10 + 8 = 18
            s1_channel = self.s1_training.shape[3]
            s2_channel = self.s2_training.shape[3]
            shape = (input_width, input_height, s1_channel + s2_channel)

        elif channel == 's2_rgb':
            # channel number should be fixed as 3 (r,g,b)
            shape = (input_width, input_height, 3)

        return shape

    #===================================================================
    # Training-data Generator
    #===================================================================
    def trainGenerator(self, batch_size=32, channel='full', train_mode='real'):

        if train_mode == 'test':
            batch_size = 32

        train_size = self.get_train_size(train_mode)
        print(train_size)

        while True:
            for i in range(0, train_size, batch_size):
                start_pos = i
                end_pos = min(i + batch_size, train_size)

                if channel == 'full':
                    train_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos])
                    train_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                    train_y_batch = np.asarray(self.label_training[start_pos:end_pos])
                    # concatenate s1 and s2 data along the last axis
                    train_concat_X_batch = np.concatenate([train_s1_X_batch, train_s2_X_batch], axis=-1)
                    # According to "fit_generator" on Keras.io, the output from the generator must
                    # be a tuple (inputs, targets), thus,
                    yield (train_concat_X_batch, train_y_batch)

                elif channel == 's2_rgb':
                    # sentinel-2 first 3 channels refer to B, G, R of an image
                    # get channel 2, 1, 0 as rgb
                    train_s2_rgb_X_batch = np.asarray(self.s2_training[start_pos:end_pos][...,2::-1])
                    train_y_batch = np.asarray(self.label_training[start_pos:end_pos])
                    yield (train_s2_rgb_X_batch, train_y_batch)


    #===================================================================
    # Validation-data Generator
    #===================================================================
    def valGenerator(self, batch_size=32, channel='full', train_mode='real'):

        if train_mode == 'test':
            batch_size = 32

        val_size = self.get_val_size(train_mode)
        print(val_size)

        while True:
            # Generate data with batch_size
            for i in range(0, val_size, batch_size):
                start_pos = i
                end_pos = min(i + batch_size, val_size)

                if channel == 'full':
                    val_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos])
                    val_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
                    val_y_batch = np.asarray(self.label_validation[start_pos:end_pos])
                    # concatenate s1 and s2 data along the last axis
                    val_concat_X_batch = np.concatenate([val_s1_X_batch, val_s2_X_batch], axis=-1)
                    # According to "fit_generator" on Keras.io, the output from the generator must
                    # be a tuple (inputs, targets), thus,
                    yield (val_concat_X_batch, val_y_batch)

                elif channel == 's2_rgb':
                    # sentinel-2 first 3 channels refer to B, G, R of an image
                    # get channel 2, 1, 0 as rgb
                    val_s2_rgb_X_batch = np.asarray(self.s2_validation[start_pos:end_pos][...,2::-1])
                    val_y_batch = np.asarray(self.label_validation[start_pos:end_pos])
                    yield (val_s2_rgb_X_batch, val_y_batch)


    #===================================================================
    # Balanced Training-data Generator
    #===================================================================
    def balancedTrainGenerator(self, batch_size=32, channel='full', train_mode='real'):

        if train_mode == 'test':
            batch_size = 32

        train_size = self.get_balanced_train_size(train_mode)
        print(train_size)

        # Generate data with batch_size
        while True:
            for i in range(0, train_size, batch_size):
                start_pos = i
                end_pos = min(i + batch_size, train_size)

                if channel == 'full':
                    s1_tmp, s2_tmp, y_tmp = [], [], []
                    for p in range(start_pos, end_pos):
                        idx = self.balanced_train_idx_list[p]
                        if idx >= self.train_size:
                            s1_tmp.append(self.s1_validation[idx - self.train_size])
                            s2_tmp.append(self.s2_validation[idx - self.train_size])
                            y_tmp.append(self.label_validation[idx - self.train_size]) 
                        else:
                            s1_tmp.append(self.s1_training[idx])
                            s2_tmp.append(self.s2_training[idx])
                            y_tmp.append(self.label_training[idx])

                    train_s1_X_batch = np.asarray(s1_tmp)
                    train_s2_X_batch = np.asarray(s2_tmp)
                    train_y_batch = np.asarray(y_tmp)
                    # concatenate s1 and s2 data along the last axis
                    train_concat_X_batch = np.concatenate([train_s1_X_batch, train_s2_X_batch], axis=-1)
                    # According to "fit_generator" on Keras.io, the output from the generator must
                    # be a tuple (inputs, targets), thus,
                    yield (train_concat_X_batch, train_y_batch)

                elif channel == 's2_rgb':
                    s2_tmp, y_tmp = [], []
                    for p in range(start_pos, end_pos):
                        idx = self.balanced_train_idx_list[p]
                        if idx >= self.train_size:
                            # get channel 2, 1, 0 as rgb
                            s2_tmp.append(self.s2_validation[idx - self.train_size][...,2::-1])
                            y_tmp.append(self.label_validation[idx - self.train_size]) 
                        else:
                            s2_tmp.append(self.s2_training[idx][...,2::-1])
                            y_tmp.append(self.label_training[idx])

                    train_s2_rgb_X_batch = np.asarray(s2_tmp)
                    train_y_batch = np.asarray(y_tmp)
                    # According to "fit_generator" on Keras.io, the output from the generator must
                    # be a tuple (inputs, targets), thus,
                    yield (train_s2_rgb_X_batch, train_y_batch)


    #===================================================================
    # Balanced Validation-data Generator
    #===================================================================
    def balancedValGenerator(self, batch_size=32, channel='s2_rgb', train_mode='real'):
        
        if train_mode == 'test':
            batch_size = 32

        val_size = self.get_balanced_val_size(train_mode)
        print(val_size)

        while True:
            # Generate data with batch_size
            for i in range(0, val_size, batch_size):
                start_pos = i
                end_pos = min(i + batch_size, val_size)

                if channel == 'full':
                    s1_tmp, s2_tmp, y_tmp = [], [], []
                    for p in range(start_pos, end_pos):
                        idx = self.balanced_val_idx_list[p]
                        if idx >= len(self.label_training):
                            s1_tmp.append(self.s1_validation[idx - self.train_size])
                            s2_tmp.append(self.s2_validation[idx - self.train_size])
                            y_tmp.append(self.label_validation[idx - self.train_size])
                        else:
                            s1_tmp.append(self.s1_training[idx])
                            s2_tmp.append(self.s2_training[idx])
                            y_tmp.append(self.label_training[idx])

                    val_s1_X_batch = np.asarray(s1_tmp)
                    val_s2_X_batch = np.asarray(s2_tmp)
                    val_y_batch = np.asarray(y_tmp)
                    # concatenate s1 and s2 data along the last axis
                    val_concat_X_batch = np.concatenate([val_s1_X_batch, val_s2_X_batch], axis=-1)
                    # According to "fit_generator" on Keras.io, the output from the generator must
                    # be a tuple (inputs, targets), thus,
                    yield (val_concat_X_batch, val_y_batch)

                elif channel == 's2_rgb':
                    s2_tmp, y_tmp = [], []
                    for p in range(start_pos, end_pos):
                        idx = self.balanced_val_idx_list[p]
                        if idx >= len(self.label_training):
                            # get channel 2, 1, 0 as rgb
                            s2_tmp.append(self.s2_validation[idx - self.train_size][...,2::-1])
                            y_tmp.append(self.label_validation[idx - self.train_size])
                        else:
                            s2_tmp.append(self.s2_training[idx][...,2::-1])
                            y_tmp.append(self.label_training[idx])

                    val_s2_rgb_X_batch = np.asarray(s2_tmp)
                    val_y_batch = np.asarray(y_tmp)
                    # According to "fit_generator" on Keras.io, the output from the generator must
                    # be a tuple (inputs, targets), thus,
                    yield (val_s2_rgb_X_batch, val_y_batch)


    #===================================================================
    # Get size methods based on 'train_mode'
    #===================================================================
    def get_train_size(self, train_mode):
        if train_mode == 'real':
            return self.train_size
        elif train_mode == 'test':
            return self.small_train_size

    def get_val_size(self, train_mode):
        if train_mode == 'real':
            return self.val_size
        elif train_mode == 'test':
            return self.small_val_size

    def get_balanced_train_size(self, train_mode):
        if train_mode == 'real':
            return self.balanced_train_size
        elif train_mode == 'test':
            return self.small_train_size

    def get_balanced_val_size(self, train_mode):
        if train_mode == 'real':
            return self.balanced_val_size
        elif train_mode == 'test':

