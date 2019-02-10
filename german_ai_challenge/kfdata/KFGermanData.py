#KFGermanData.py
# KF 12/17/2018
# KF 12/26/2018 Update
# KF 01/03/2019 - add s1_ch56 mode
# KF 01/04/2018   change s1_ch56 to s1_ch78
#                 change s1_ch78 to s1_ch5678

import os
import h5py
import numpy as np
import random

class GermanData:

    #####################################################################
    #                          Initialization
    #####################################################################
    def __init__(self, params):

        #================================================================
        # params - paramter dictionary
        #   parameter key list:
        #       base_dir              - Base directory of the dataset
        #       train_filename        - train data file name
        #       val_filename          - validation data file name
        #       round1_test_filename  - round 1 test data file name
        #       train_mode            - either 'real' or 'test'
        #       batch_size            - batch size
        #       data_channel          - 'full', 's2_rgb'
        #       data_gen_mode         - either 'original' or 'balanced'
        #       data_normalize        - either 'yes' or 'no'
        #================================================================

        # Initialize paths parameters
        self.base_dir = params.get('base_dir')
        self.train_filename = params.get('train_filename')
        self.val_filename = params.get('val_filename')
        self.kf_data_filename = params.get('kf_data_filename')
        self.kf_val_filename = params.get('kf_val_filename')
        self.kf_test2A_filename = params.get('kf_test2A_filename')
        self.kf_test2B_filename = params.get('kf_test2B_filename')

        self.round1_testA_filename = params.get('round1_testA_filename')
        self.round1_testB_filename = params.get('round1_testB_filename')
        self.round2_testA_filename = params.get('round2_testA_filename')
        #self.round2_testB_filename = params.get('round2_testB_filename')
        # Initialize other parameters
        self.train_mode = params.get('train_mode')
        self.batch_size = params.get('batch_size')
        self.data_channel = params.get('data_channel')
        self.data_gen_mode = params.get('data_gen_mode')
        self.data_normalize = params.get('data_normalize')

        self.path_round1_testA = os.path.join(self.base_dir, self.round1_testA_filename)
        self.path_round1_testB = os.path.join(self.base_dir, self.round1_testB_filename)

        # KF 01/31/2019
        # Change train/val/test dataset to kf - preprocessed source
        if self.data_gen_mode in ['kf', 'kf_data_only']:
            self.path_training = os.path.join(self.base_dir, self.kf_data_filename)
            self.path_validation = os.path.join(self.base_dir, self.kf_val_filename) 
            self.path_round2_testA = os.path.join(self.base_dir, self.kf_test2A_filename) 
            #self.path_round2_testB = os.path.join(self.base_dir, self.kf_test2B_filename) 
            print('='*80)
            print('Processed Datasets are Used!!')
            print('='*80)
            print('Training Data   :', self.kf_data_filename)
            print('Validation Data :', self.kf_val_filename)
            print('Test 2A Data    :', self.kf_test2A_filename)
            print('Test 2B Data    :', self.kf_test2B_filename)
        else:
            self.path_training = os.path.join(self.base_dir, self.train_filename)
            self.path_validation = os.path.join(self.base_dir, self.val_filename) 
            self.path_round2_testA = os.path.join(self.base_dir, self.round2_testA_filename)
            #self.path_round2_testB = os.path.join(self.base_dir, self.round2_testB_filename)

        #----------------------------------------------------------------
        # 1. Paths Validation
        #----------------------------------------------------------------
        #self.print_title('Load Data')
        #print("[KF INFO] Validate data file existance: ")
        #for f in [self.train_filename, 
        #            self.val_filename, 
        #            self.kf_data_filename,
        #            self.kf_val_filename,
        #            self.round1_testA_filename, 
        #            self.round1_testB_filename,
        #            self.round2_testA_filename]:
        #    if f in os.listdir(self.base_dir):
        #        print('|--CHECK!-->', f)
        #print('')

        #----------------------------------------------------------------
        # 2. Load Data
        #----------------------------------------------------------------
        # Train/Val Data
        fid_training = h5py.File(self.path_training, 'r')
        fid_validation = h5py.File(self.path_validation, 'r')
        self.s1_training = fid_training['sen1']
        self.s2_training = fid_training['sen2']
        self.label_training = fid_training['label']
        self.s1_validation = fid_validation['sen1']
        self.s2_validation = fid_validation['sen2']
        self.label_validation = fid_validation['label']

        # Test Data
        fid_test1A = h5py.File(self.path_round1_testA, 'r')
        fid_test1B = h5py.File(self.path_round1_testB, 'r')
        fid_test2A = h5py.File(self.path_round2_testA, 'r')
        #fid_test2B = h5py.File(self.path_round2_testB, 'r')

        self.s1_test1A = fid_test1A['sen1']
        self.s2_test1A = fid_test1A['sen2']
        self.s1_test1B = fid_test1B['sen1']
        self.s2_test1B = fid_test1B['sen2']
        self.s1_test2A = fid_test2A['sen1']
        self.s2_test2A = fid_test2A['sen2']
        #self.s1_test2B = fid_test2B['sen1']
        #self.s2_test2B = fid_test2B['sen2']
        print("[KF INFO] Data loaded successfully!")
        # Show data information
        self.showDataInfo()

        #----------------------------------------------------------------
        # 3. Save data dimension
        #----------------------------------------------------------------
        # For External Use:
        self.data_dimension = self.getDataDimension()

        #----------------------------------------------------------------
        # 4. Prepare Data and Generator
        #----------------------------------------------------------------
        if self.data_gen_mode == 'original':
        
            self.train_size = len(self.label_training)
            self.val_size = len(self.label_validation)

            # Reset parameters for 'test' mode
            if self.train_mode == 'test':
                self.batch_size = 32
                self.train_size = 4000
                self.val_size = 1000
            
            # Configure data generator
            # For External Use:
            self.train_gen = self.trainGenerator()
            self.val_gen = self.valGenerator()

        # KF 01/31/2019
        elif self.data_gen_mode == 'kf':
            self.train_size = len(self.label_training)
            self.val_size = len(self.label_validation)
            self.train_gen = self.trainGenerator()
            self.val_gen = self.valGenerator()

        # KF 02/01/2019
        elif self.data_gen_mode == 'kf_data_only':
            self.print_title('Use kf_data ONLY!')
            # Split for training and validation set
            #self.val_split_idx = int(np.ceil(self.label_training.shape[0] * 4 / 5))
            self.val_split_idx = int(np.ceil(self.label_training.shape[0] * 9 / 10))
            self.train_size = self.val_split_idx
            self.val_size = self.label_training.shape[0] - self.train_size
            self.train_gen = self.trainGenerator()
            self.val_gen = self.valGenerator()

        elif self.data_gen_mode == 'balanced':
            # prepare balanced data 
            self.useBalancedData()

        elif self.data_gen_mode == 'val_dataset_only':
            self.useValidationDataset()

        elif self.data_gen_mode == 'shuffled_original':
            self.useShuffledOriginalData()
            

        #----------------------------------------------------------------
        # 5. Prepare Round1 Test data for prediction
        #----------------------------------------------------------------
        #self.test_data = self.getTestData()

        #----------------------------------------------------------------
        # 6. Get data class weight
        #----------------------------------------------------------------
        self.label_qty = np.sum(self.label_training, axis=0)
        self.label_qty += np.sum(self.label_validation, axis=0)
        self.class_weight = (self.label_training.shape[0] + self.label_validation.shape[0])/self.label_qty
        #self.class_weight /= min(self.class_weight)
        self.class_weight /= max(self.class_weight)
        

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
        #print("Round1 TestA data shapes:")
        #print('  Sentinel 1 data shape :', self.s1_test1A.shape)
        #print('  Sentinel 2 data shape :', self.s2_test1A.shape)
        #print('-'*65)
        #print("Round1 TestB data shapes:")
        #print('  Sentinel 1 data shape :', self.s1_test1B.shape)
        #print('  Sentinel 2 data shape :', self.s2_test1B.shape)
        print("Round2 TestA data shapes:")
        print('  Sentinel 1 data shape :', self.s1_test2A.shape)
        print('  Sentinel 2 data shape :', self.s2_test2A.shape)
        #print('-'*65)
        #print("Round1 TestB data shapes:")
        #print('  Sentinel 1 data shape :', self.s1_test1B.shape)
        #print('  Sentinel 2 data shape :', self.s2_test1B.shape)


    #===================================================================
    # Get balance data :
    #   Save balanced data as class variables
    #===================================================================
    def useBalancedData(self):
        self.print_title('Get Balanced Data')
        label_all = np.concatenate((self.label_training, self.label_validation), axis=0)
        label_qty = np.sum(label_all, axis=0)
        min_size = int(np.min(label_qty))
        print("Minimal sample size per class : ", min_size)

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
            if idx >= len(self.label_training):
                label_tmp = self.label_validation[idx - len(self.label_training)]
            else:
                label_tmp = self.label_training[idx]
            test_res += label_tmp
        for i, num in enumerate(test_res):
            print("  number of samples in class %2d :" % i, int(num))

        # Shuffle inplace:
        random.shuffle(sorted_balanced_idx_list)
        # Divide training and validation dataset
        split = len(sorted_balanced_idx_list) * 4 // 5

        # Save to variables
        self.balanced_train_idx_list = sorted_balanced_idx_list[:split]
        self.balanced_val_idx_list = sorted_balanced_idx_list[split:]
        self.train_size = len(self.balanced_train_idx_list)
        self.val_size = len(self.balanced_val_idx_list)


        # Reset parameters for 'test' mode
        if self.train_mode == 'test':
            self.batch_size = 32
            self.train_size = 4000
            self.val_size = 1000
        
        # Configure data generator
        # For External Use
        self.train_gen = self.balancedTrainGenerator()
        self.val_gen = self.balancedValGenerator()

    #===================================================================
    # Use Shuffled Original Dataset 
    # KF 01/07/2019
    # KF 01/14/2019 - Major update for improving data gen speed
    #===================================================================
    def useShuffledOriginalData(self):
        self.print_title('Get Shuffled Original Data')
        # Set train size and validation size as the same as the original data
        self.train_size = len(self.label_training)
        self.val_size = len(self.label_validation)

        # --------------------------------------------------------------
        # KF 01/14/2019
        # Ignore Test Mode !!!
        # --------------------------------------------------------------
        # Reset parameters for 'test' mode
        #if self.train_mode == 'test':
        #    self.batch_size = 32
        #    self.train_size = 4000
        #    self.val_size = 1000

        # --------------------------------------------------------------
        # KF 01/07/2019
        # Shuffle all index of both the training and validation dataset
        # Low efficiency
        # --------------------------------------------------------------
        #self.index = np.arange(self.train_size + self.val_size)
        ## Bookkeeping
        #random.seed(2019)

        ## shuffle the index inplace
        #random.shuffle(self.index)

        ## Save both index lists for training and validation
        #self.shuffled_train_idx_list = self.index[:self.train_size]
        #self.shuffled_val_idx_list = self.index[self.train_size:]

        # --------------------------------------------------------------
        # KF 01/07/2019
        # Update: 
        #   1. Based on self.batch_size, get all indices of the first item
        #      in each batch into a list 'self.batch_head_idx'
        #   2. Shuffle the self.batch_head_idx
        #   3. split the shuffled index list into 'shuffled_train_batch_head_idx'.
        #      and 'shuffled_val_batch_head_idx'
        # --------------------------------------------------------------
        # Add train_dataset batch index head
        batch_head_idx = [i for i in range(self.train_size) if i % self.batch_size == 0]
        # Save train length
        train_len = len(batch_head_idx)
        # Append val_dataset batch head index 
        batch_head_idx += [i + self.train_size for i in range(self.val_size) if i % self.batch_size == 0]

        # Set random seed:
        random.seed(2019)
        # shuffle the batch head index list
        random.shuffle(batch_head_idx)
        
        # Split train indices and val indices
        self.shuffled_train_batch_head_idx_list = batch_head_idx[:train_len]
        self.shuffled_val_batch_head_idx_list = batch_head_idx[train_len:]
        
        # Reset parameters for 'test' mode
        if self.train_mode == 'test':
            self.batch_size = 32
            self.train_size = 4000
            self.val_size = 1000
        # --------------------------------------------------------------
        # Configure data generator
        # --------------------------------------------------------------
        # For External Use
        self.train_gen = self.shuffledOriginalTrainGenerator()
        self.val_gen = self.shuffledOriginalValGenerator()
            
    #===================================================================
    # Use Validation dataset ONLY. (for both training and validation)
    # KF 12/26/2018
    #===================================================================
    def useValidationDataset(self):
        self.print_title('Use validation dataset ONLY!')
        # Split for training and validation set
        self.val_split_idx = int(np.ceil(self.label_validation.shape[0] * 4 / 5))
        self.train_size = self.val_split_idx
        self.val_size = self.label_validation.shape[0] - self.train_size

        # Normalize
        # KF 01/29/2019
        if self.data_normalize == 'yes':
            self.print_title('Replace Outliers & Standardization')
            # Make sure the data is loaded into memory
            print(type(self.s1_validation))
            print(type(self.s2_validation))
            print(type(self.s1_test2A))
            print(type(self.s2_test2A))
            self.s1_validation = self.s1_validation[:]
            self.s2_validation = self.s2_validation[:]
            self.s1_test2A = self.s1_test2A[:]
            self.s2_test2A = self.s2_test2A[:]
            print(type(self.s1_validation))
            print(type(self.s2_validation))
            print(type(self.s1_test2A))
            print(type(self.s2_test2A))
            # rejecting threshold for outliers
            m = 2.0
            # S1
            for ch in [4,5,6,7]:
                data_all = np.concatenate([self.s1_validation[..., ch], self.s1_test2A[..., ch]], axis=0)
                med = np.median(data_all)
                std = np.std(data_all)
                print('-'*65)
                print('S1 - Ch:', ch)
                print('-'*65)
                print('data_all shape:', data_all.shape)

                print('Replace Outliers ...')
                for (n, w, h), value in np.ndenumerate(self.s1_validation[..., ch]):
                    if self.s1_validation[n, w, h, ch] - med > m * std:
                        self.s1_validation[n, w, h, ch] = m * std
                    elif self.s1_validation[n, w, h, ch] - med < - m * std:
                        self.s1_validation[n, w, h, ch] = - m * std
                for (n, w, h), value in np.ndenumerate(self.s1_test2A[..., ch]):
                    if self.s1_test2A[n, w, h, ch] - med > m * std:
                        self.s1_test2A[n, w, h, ch] = m * std
                    elif self.s1_test2A[n, w, h, ch] - med < - m * std:
                        self.s1_test2A[n, w, h, ch] = - m * std
                #for i in range(len(self.s1_validation)):
                #    norm = np.linalg.norm(self.s1_validation[i,:,:,ch])
                #    self.s1_validation[i,:,:,ch] /= norm
                # Rescale to maximum 1.0
                #self.s1_validation[...,ch] /= np.max(self.s1_validation[..., ch])
                # Standardize
                # recalculate the std !!!
                data_all = np.concatenate([self.s1_validation[..., ch], self.s1_test2A[..., ch]], axis=0)
                std = np.std(data_all)
                self.s1_validation[...,ch] /= std
                self.s1_test2A[...,ch] /= std

                print('Global std after outliers replaced:', std)
                print('[train/val data]')
                print('Min    :', np.min(self.s1_validation[..., ch]))
                print('Max    :', np.max(self.s1_validation[..., ch]))
                print('Mean   :', np.mean(self.s1_validation[..., ch]))
                print('Median :', np.median(self.s1_validation[..., ch]))
                print('Std    :', np.std(self.s1_validation[..., ch]))
                print('[test2A]')
                print('Min    :', np.min(self.s1_test2A[..., ch]))
                print('Max    :', np.max(self.s1_test2A[..., ch]))
                print('Mean   :', np.mean(self.s1_test2A[..., ch]))
                print('Median :', np.median(self.s1_test2A[..., ch]))
                print('Std    :', np.std(self.s1_test2A[..., ch]))
            
            # S2
            for ch in range(10):
                data_all = np.concatenate([self.s2_validation[..., ch], self.s2_test2A[..., ch]], axis=0)
                std = np.std(data_all)
                print('-'*65)
                print('S2 - Ch:', ch)
                print('-'*65)
                print('data_all shape:', data_all.shape)
                #for i in range(len(self.s2_validation)):
                #    norm = np.linalg.norm(self.s2_validation[i,:,:,ch])
                #    self.s2_validation[i,:,:,ch] /= norm
                # Rescale to maximum 1.0
                #self.s2_validation[...,ch] /= np.max(self.s2_validation[..., ch])
                # Standardize
                # recalculate the std !!!
                data_all = np.concatenate([self.s2_validation[..., ch], self.s2_test2A[..., ch]], axis=0)
                std = np.std(data_all)
                self.s2_validation[..., ch] /= std
                self.s2_test2A[..., ch] /= std

                print('Global std after outliers replaced:', std)
                print('[train/val data]')
                print('Min    :', np.min(self.s2_validation[..., ch]))
                print('Max    :', np.max(self.s2_validation[..., ch]))
                print('Mean   :', np.mean(self.s2_validation[..., ch]))
                print('Median :', np.median(self.s2_validation[..., ch]))
                print('Std    :', np.std(self.s2_validation[..., ch]))
                print('[test2A]') 
                print('Min    :', np.min(self.s2_test2A[..., ch]))
                print('Max    :', np.max(self.s2_test2A[..., ch]))
                print('Mean   :', np.mean(self.s2_test2A[..., ch]))
                print('Median :', np.median(self.s2_test2A[..., ch]))
                print('Std    :', np.std(self.s2_test2A[..., ch]))
            

        self.train_gen = self.trainGenerator()
        self.val_gen = self.valGenerator()

    #===================================================================
    # Get data shape
    #===================================================================
    def getDataDimension(self):

        input_width = self.s1_training.shape[1] 
        input_height = self.s1_training.shape[2]

        if self.data_channel == 'full':
            # channel number should be 10 + 8 = 18
            s1_channel = self.s1_training.shape[3]
            s2_channel = self.s2_training.shape[3]
            dimension = (input_width, input_height, s1_channel + s2_channel)

        elif self.data_channel == 's2_rgb':
            # channel number should be fixed as 3 (r,g,b)
            dimension = (input_width, input_height, 3)

        elif self.data_channel == 's1':
            # channel number should be 10 + 8 = 18
            s1_channel = self.s1_training.shape[3]
            dimension = (input_width, input_height, s1_channel)

        elif self.data_channel == 's2':
            # channel number should be 10 + 8 = 18
            s2_channel = self.s2_training.shape[3]
            dimension = (input_width, input_height, s2_channel)
            
        elif self.data_channel == 's1_ch5678':
            dimension = (input_width, input_height, 4)

        elif self.data_channel == 's1_ch5678+s2':
            dimension = (input_width, input_height, 4 + self.s2_training.shape[3])

        return dimension

    #####################################################################
    #                         Data Generators
    #####################################################################

    #===================================================================
    # Training-data Generator
    #===================================================================
    def trainGenerator(self):

        train_size = self.train_size
        batch_size = self.batch_size
        channel = self.data_channel
        print("")
        print("Train size:", train_size, "Batch size:", batch_size, "Channel:", channel)
        
        while True:
            for i in range(0, train_size, batch_size):

                if self.data_gen_mode == 'val_dataset_only':

                    start_pos = i
                    end_pos = min(i + batch_size, train_size)

                    train_y_batch = np.asarray(self.label_validation[start_pos:end_pos])

                    if channel == 'full':
                        train_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos])
                        train_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
                        # concatenate s1 and s2 data along the last axis
                        train_concat_X_batch = np.concatenate([train_s1_X_batch, train_s2_X_batch], axis=-1)
                        # According to "fit_generator" on Keras.io, the output from the generator must
                        # be a tuple (inputs, targets), thus,
                        yield (train_concat_X_batch, train_y_batch)

                    elif channel == 's2_rgb':
                        # sentinel-2 first 3 channels refer to B, G, R of an image
                        # get channel 2, 1, 0 as rgb
                        train_s2_rgb_X_batch = np.asarray(self.s2_validation[start_pos:end_pos][...,2::-1])
                        yield (train_s2_rgb_X_batch, train_y_batch)
                        
                    elif channel == 's1':
                        train_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos])
                        yield (train_s1_X_batch, train_y_batch)

                    elif channel == 's2':
                        train_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
                        yield (train_s2_X_batch, train_y_batch)

                    elif channel == 's1_ch5678':
                        train_X_batch = np.asarray(self.s1_validation[start_pos:end_pos][...,4:])
                        yield (train_X_batch, train_y_batch)

                    elif channel == 's1_ch5678+s2':
                        train_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos][...,4:])
                        train_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
                        train_concat_X_batch = np.concatenate([train_s1_X_batch, train_s2_X_batch], axis=-1)
                        yield (train_concat_X_batch, train_y_batch)

                # KF 02/01/2019
                elif self.data_gen_mode == 'kf_data_only':
                    #print('[KF INFO] Train Data Gen - kf_data ONLY!!!')
                    start_pos = i
                    end_pos = min(i + batch_size, train_size)

                    train_y_batch = np.asarray(self.label_training[start_pos:end_pos])

                    if channel == 'full':
                        train_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos])
                        train_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                        # concatenate s1 and s2 data along the last axis
                        train_concat_X_batch = np.concatenate([train_s1_X_batch, train_s2_X_batch], axis=-1)
                        # According to "fit_generator" on Keras.io, the output from the generator must
                        # be a tuple (inputs, targets), thus,
                        yield (train_concat_X_batch, train_y_batch)

                    elif channel == 's2_rgb':
                        # sentinel-2 first 3 channels refer to B, G, R of an image
                        # get channel 2, 1, 0 as rgb
                        train_s2_rgb_X_batch = np.asarray(self.s2_training[start_pos:end_pos][...,2::-1])
                        yield (train_s2_rgb_X_batch, train_y_batch)
                        
                    elif channel == 's1':
                        train_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos])
                        yield (train_s1_X_batch, train_y_batch)

                    elif channel == 's2':
                        train_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                        yield (train_s2_X_batch, train_y_batch)

                    elif channel == 's1_ch5678':
                        train_X_batch = np.asarray(self.s1_training[start_pos:end_pos][...,4:])
                        yield (train_X_batch, train_y_batch)

                    elif channel == 's1_ch5678+s2':
                        train_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos][...,4:])
                        train_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                        train_concat_X_batch = np.concatenate([train_s1_X_batch, train_s2_X_batch], axis=-1)
                        yield (train_concat_X_batch, train_y_batch)

                else:
                    start_pos = i
                    end_pos = min(i + batch_size, train_size)

                    train_y_batch = np.asarray(self.label_training[start_pos:end_pos])

                    if channel == 'full':
                        train_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos])
                        train_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                        # concatenate s1 and s2 data along the last axis
                        train_concat_X_batch = np.concatenate([train_s1_X_batch, train_s2_X_batch], axis=-1)
                        # According to "fit_generator" on Keras.io, the output from the generator must
                        # be a tuple (inputs, targets), thus,
                        yield (train_concat_X_batch, train_y_batch)

                    elif channel == 's2_rgb':
                        # sentinel-2 first 3 channels refer to B, G, R of an image
                        # get channel 2, 1, 0 as rgb
                        train_s2_rgb_X_batch = np.asarray(self.s2_training[start_pos:end_pos][...,2::-1])
                        yield (train_s2_rgb_X_batch, train_y_batch)
                        
                    elif channel == 's1':
                        train_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos])
                        yield (train_s1_X_batch, train_y_batch)

                    elif channel == 's2':
                        train_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                        yield (train_s2_X_batch, train_y_batch)

                    elif channel == 's1_ch5678':
                        train_X_batch = np.asarray(self.s1_training[start_pos:end_pos][...,4:])
                        yield (train_X_batch, train_y_batch)
                    elif channel == 's1_ch5678+s2':
                        train_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos][...,4:])
                        train_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                        train_concat_X_batch = np.concatenate([train_s1_X_batch, train_s2_X_batch], axis=-1)
                        yield (train_concat_X_batch, train_y_batch)

    #===================================================================
    # Validation-data Generator
    #===================================================================
    def valGenerator(self):

        val_size = self.val_size
        batch_size = self.batch_size
        channel = self.data_channel
        print("")
        print("Validation size:", val_size, "Batch size:", batch_size, "Channel:", channel)

        while True:
            for i in range(0, val_size, batch_size):

                if self.data_gen_mode == 'val_dataset_only':

                    # Generate data with batch_size
                    start_pos = i + self.val_split_idx
                    end_pos = min(i + batch_size, val_size) + self.val_split_idx

                    val_y_batch = np.asarray(self.label_validation[start_pos:end_pos])

                    if channel == 'full':
                        val_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos])
                        val_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
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

                    elif channel == 's1':
                        val_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos])
                        yield (val_s1_X_batch, val_y_batch)

                    elif channel == 's2':
                        val_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
                        yield (val_s2_X_batch, val_y_batch)
            
                    elif channel == 's1_ch5678':
                        val_X_batch = np.asarray(self.s1_validation[start_pos:end_pos][...,4:])
                        val_y_batch = np.asarray(self.label_validation[start_pos:end_pos])
                        yield (val_X_batch, val_y_batch)

                    elif channel == 's1_ch5678+s2':
                        val_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos][...,4:])
                        val_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
                        val_concat_X_batch = np.concatenate([val_s1_X_batch, val_s2_X_batch], axis=-1)
                        yield (val_concat_X_batch, val_y_batch)

                # KF 02/01/2019
                elif self.data_gen_mode == 'kf_data_only':
                    #print('[KF INFO] Val Data Gen - kf_data ONLY!!!')
                    # Generate data with batch_size
                    start_pos = i + self.val_split_idx
                    end_pos = min(i + batch_size, val_size) + self.val_split_idx

                    val_y_batch = np.asarray(self.label_training[start_pos:end_pos])

                    if channel == 'full':
                        val_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos])
                        val_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                        # concatenate s1 and s2 data along the last axis
                        val_concat_X_batch = np.concatenate([val_s1_X_batch, val_s2_X_batch], axis=-1)
                        # According to "fit_generator" on Keras.io, the output from the generator must
                        # be a tuple (inputs, targets), thus,
                        yield (val_concat_X_batch, val_y_batch)

                    elif channel == 's2_rgb':
                        # sentinel-2 first 3 channels refer to B, G, R of an image
                        # get channel 2, 1, 0 as rgb
                        val_s2_rgb_X_batch = np.asarray(self.s2_training[start_pos:end_pos][...,2::-1])
                        val_y_batch = np.asarray(self.label_training[start_pos:end_pos])
                        yield (val_s2_rgb_X_batch, val_y_batch)

                    elif channel == 's1':
                        val_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos])
                        yield (val_s1_X_batch, val_y_batch)

                    elif channel == 's2':
                        val_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                        yield (val_s2_X_batch, val_y_batch)
            
                    elif channel == 's1_ch5678':
                        val_X_batch = np.asarray(self.s1_training[start_pos:end_pos][...,4:])
                        val_y_batch = np.asarray(self.label_training[start_pos:end_pos])
                        yield (val_X_batch, val_y_batch)

                    elif channel == 's1_ch5678+s2':
                        val_s1_X_batch = np.asarray(self.s1_training[start_pos:end_pos][...,4:])
                        val_s2_X_batch = np.asarray(self.s2_training[start_pos:end_pos])
                        val_concat_X_batch = np.concatenate([val_s1_X_batch, val_s2_X_batch], axis=-1)
                        yield (val_concat_X_batch, val_y_batch)

                else: 
                    start_pos = i
                    end_pos = min(i + batch_size, val_size)

                    val_y_batch = np.asarray(self.label_validation[start_pos:end_pos])

                    if channel == 'full':
                        val_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos])
                        val_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
                        # concatenate s1 and s2 data along the last axis
                        val_concat_X_batch = np.concatenate([val_s1_X_batch, val_s2_X_batch], axis=-1)
                        # According to "fit_generator" on Keras.io, the output from the generator must
                        # be a tuple (inputs, targets), thus,
                        yield (val_concat_X_batch, val_y_batch)

                    elif channel == 's2_rgb':
                        # sentinel-2 first 3 channels refer to B, G, R of an image
                        # get channel 2, 1, 0 as rgb
                        val_s2_rgb_X_batch = np.asarray(self.s2_validation[start_pos:end_pos][...,2::-1])
                        yield (val_s2_rgb_X_batch, val_y_batch)

                    elif channel == 's1':
                        val_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos])
                        yield (val_s1_X_batch, val_y_batch)

                    elif channel == 's2':
                        val_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
                        yield (val_s2_X_batch, val_y_batch)

                    elif channel == 's1_ch5678':
                        val_X_batch = np.asarray(self.s1_validation[start_pos:end_pos][...,4:])
                        val_y_batch = np.asarray(self.label_validation[start_pos:end_pos])
                        yield (val_X_batch, val_y_batch)

                    elif channel == 's1_ch5678+s2':
                        val_s1_X_batch = np.asarray(self.s1_validation[start_pos:end_pos][...,4:])
                        val_s2_X_batch = np.asarray(self.s2_validation[start_pos:end_pos])
                        val_concat_X_batch = np.concatenate([val_s1_X_batch, val_s2_X_batch], axis=-1)
                        yield (val_concat_X_batch, val_y_batch)

    #===================================================================
    # Balanced Training-data Generator
    #===================================================================
    def balancedTrainGenerator(self):

        train_size = self.train_size
        batch_size = self.batch_size
        channel = self.data_channel
        print("")
        print("Train size:", train_size, "Batch size:", batch_size, "Channel:", channel)

        # Generate data with batch_size
        while True:
            for i in range(0, train_size, batch_size):
                start_pos = i
                end_pos = min(i + batch_size, train_size)

                if channel == 'full':
                    s1_tmp, s2_tmp, y_tmp = [], [], []
                    for p in range(start_pos, end_pos):
                        idx = self.balanced_train_idx_list[p]
                        if idx >= len(self.label_training):
                            s1_tmp.append(self.s1_validation[idx -len(self.label_training)])
                            s2_tmp.append(self.s2_validation[idx - len(self.label_training)])
                            y_tmp.append(self.label_validation[idx - len(self.label_training)])
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
                        if idx >= len(self.label_training):
                            # get channel 2, 1, 0 as rgb
                            s2_tmp.append(self.s2_validation[idx -len(self.label_training)][...,2::-1])
                            y_tmp.append(self.label_validation[idx - len(self.label_training)])
                        else:
                            s2_tmp.append(self.s2_training[idx][...,2::-1])
                            y_tmp.append(self.label_training[idx])

                    train_s2_rgb_X_batch = np.asarray(s2_tmp)
                    train_y_batch = np.asarray(y_tmp)
                    # According to "fit_generator" on Keras.io, the output from the generator must
                    # be a tuple (inputs, targets), thus,
                    yield (train_s2_rgb_X_batch, train_y_batch)

                if channel == 's1':
                    s1_tmp, y_tmp = [], []
                    for p in range(start_pos, end_pos):
                        idx = self.balanced_train_idx_list[p]
                        if idx >= len(self.label_training):
                            s1_tmp.append(self.s1_validation[idx -len(self.label_training)])
                            y_tmp.append(self.label_validation[idx - len(self.label_training)])
                        else:
                            s1_tmp.append(self.s1_training[idx])
                            y_tmp.append(self.label_training[idx])

                    train_s1_X_batch = np.asarray(s1_tmp)
                    train_y_batch = np.asarray(y_tmp)
                    yield (train_s1_X_batch, train_y_batch)

                if channel == 's2':
                    s2_tmp, y_tmp = [], []
                    for p in range(start_pos, end_pos):
                        idx = self.balanced_train_idx_list[p]
                        if idx >= len(self.label_training):
                            s2_tmp.append(self.s2_validation[idx - len(self.label_training)])
                            y_tmp.append(self.label_validation[idx - len(self.label_training)])
                        else:
                            s2_tmp.append(self.s2_training[idx])
                            y_tmp.append(self.label_training[idx])

                    train_s2_X_batch = np.asarray(s2_tmp)
                    train_y_batch = np.asarray(y_tmp)
                    yield (train_s2_X_batch , train_y_batch)

    #===================================================================
    # Balanced Validation-data Generator
    #===================================================================
    def balancedValGenerator(self):
        
        val_size = self.val_size
        batch_size = self.batch_size
        channel = self.data_channel
        print("")
        print("Validation size:", val_size, "Batch size:", batch_size, "Channel:", channel)

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
                            s1_tmp.append(self.s1_validation[idx - len(self.label_training)])
                            s2_tmp.append(self.s2_validation[idx - len(self.label_training)])
                            y_tmp.append(self.label_validation[idx - len(self.label_training)])
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
                            s2_tmp.append(self.s2_validation[idx - len(self.label_training)][...,2::-1])
                            y_tmp.append(self.label_validation[idx - len(self.label_training)])
                        else:
                            s2_tmp.append(self.s2_training[idx][...,2::-1])
                            y_tmp.append(self.label_training[idx])

                    val_s2_rgb_X_batch = np.asarray(s2_tmp)
                    val_y_batch = np.asarray(y_tmp)
                    # According to "fit_generator" on Keras.io, the output from the generator must
                    # be a tuple (inputs, targets), thus,
                    yield (val_s2_rgb_X_batch, val_y_batch)

                if channel == 's1':
                    s1_tmp, y_tmp = [], []
                    for p in range(start_pos, end_pos):
                        idx = self.balanced_val_idx_list[p]
                        if idx >= len(self.label_training):
                            s1_tmp.append(self.s1_validation[idx - len(self.label_training)])
                            y_tmp.append(self.label_validation[idx - len(self.label_training)])
                        else:
                            s1_tmp.append(self.s1_training[idx])
                            y_tmp.append(self.label_training[idx])

                    val_s1_X_batch = np.asarray(s1_tmp)
                    val_y_batch = np.asarray(y_tmp)
                    yield (val_s1_X_batch, val_y_batch)

                if channel == 's2':
                    s2_tmp, y_tmp = [], []
                    for p in range(start_pos, end_pos):
                        idx = self.balanced_val_idx_list[p]
                        if idx >= len(self.label_training):
                            s2_tmp.append(self.s2_validation[idx - len(self.label_training)])
                            y_tmp.append(self.label_validation[idx - len(self.label_training)])
                        else:
                            s2_tmp.append(self.s2_training[idx])
                            y_tmp.append(self.label_training[idx])

                    val_s2_X_batch = np.asarray(s2_tmp)
                    val_y_batch = np.asarray(y_tmp)
                    yield (val_s2_X_batch , val_y_batch)

    #===================================================================
    # Shuffled Original Training-data Generator
    # KF 01/07/2019
    #===================================================================
    def shuffledOriginalTrainGenerator(self):
        
        train_size = self.train_size
        val_size = self.val_size
        batch_size = self.batch_size
        channel = self.data_channel
        print("Train size:", train_size, "Batch size:", batch_size, "Channel:", channel)

        #while True:
        #    for i in range(0, train_size, batch_size):
        #        start_pos = i
        #        end_pos = min(i + batch_size, train_size)

        #        if channel == 's2':
        #            s2_tmp, y_tmp = [], []
        #            for p in range(start_pos, end_pos):
        #                idx = self.shuffled_train_idx_list[p]
        #                if idx >= train_size:
        #                    s2_tmp.append(self.s2_validation[idx - train_size])
        #                    y_tmp.append(self.label_validation[idx - train_size])
        #                else:
        #                    s2_tmp.append(self.s2_training[idx])
        #                    y_tmp.append(self.label_training[idx])
        #            
        #            train_X_batch = np.asarray(s2_tmp)
        #            train_y_batch = np.asarray(y_tmp)
        #            yield (train_X_batch, train_y_batch)

        # KF - 01/14/2019
        while True:
            for i in self.shuffled_train_batch_head_idx_list:
                start_pos = i
                end_pos = i + batch_size
                # for the last batch in original train_set:
                if i < train_size and i + batch_size > train_size:
                    end_pos = train_size
                # for the last batch in original validation_set:
                if i + batch_size > train_size + val_size:
                    end_pos = train_size + val_size
                    
                if channel == 's2':
                    if start_pos < train_size:
                        train_X_batch = self.s2_training[start_pos:end_pos]
                        train_y_batch = self.label_training[start_pos:end_pos]
                    else:
                        start_pos -= train_size
                        end_pos -= train_size 
                        train_X_batch = self.s2_validation[start_pos:end_pos]
                        train_y_batch = self.label_validation[start_pos:end_pos]

                    yield (train_X_batch, train_y_batch)

                elif channel == 's1_ch5678+s2':
                    if start_pos < train_size:
                        train_X_batch = np.concatenate([self.s1_training[start_pos:end_pos][...,4:], self.s2_training[start_pos:end_pos]], axis=-1)
                        train_y_batch = self.label_training[start_pos:end_pos]
                    else:
                        start_pos -= train_size
                        end_pos -= train_size 
                        train_X_batch = np.concatenate([self.s1_validation[start_pos:end_pos][...,4:], self.s2_validation[start_pos:end_pos]], axis=-1)
                        train_y_batch = self.label_validation[start_pos:end_pos]

                    yield (train_X_batch, train_y_batch)
                    

    #===================================================================
    # Shuffled Original Validation -data Generator
    # KF 01/07/2019
    #===================================================================
    def shuffledOriginalValGenerator(self):

        train_size = self.train_size
        val_size = self.val_size
        batch_size = self.batch_size
        channel = self.data_channel
        print("Validation size:", val_size, "Batch size:", batch_size, "Channel:", channel)

        #while True:
        #    for i in range(0, val_size, batch_size):
        #        start_pos = i
        #        end_pos = min(i + batch_size, val_size)

        #        if channel == 's2':
        #            s2_tmp, y_tmp = [], []
        #            for p in range(start_pos, end_pos):
        #                idx = self.shuffled_val_idx_list[p]
        #                if idx >= train_size:
        #                    s2_tmp.append(self.s2_validation[idx - train_size])
        #                    y_tmp.append(self.label_validation[idx - train_size])
        #                else:
        #                    s2_tmp.append(self.s2_training[idx])
        #                    y_tmp.append(self.label_training[idx])
        #            
        #            val_X_batch = np.asarray(s2_tmp)
        #            val_y_batch = np.asarray(y_tmp)
        #            yield (val_X_batch, val_y_batch)

        # KF - 01/14/2019
        while True:
            for i in self.shuffled_val_batch_head_idx_list:
                start_pos = i
                end_pos = i + batch_size
                # for the last batch in original train_set:
                if i < train_size and i + batch_size > train_size:
                    end_pos = train_size
                # for the last batch in original validation_set:
                if i + batch_size > train_size + val_size:
                    end_pos = train_size + val_size
                    
                if channel == 's2':
                    if start_pos < train_size:
                        val_X_batch = self.s2_training[start_pos:end_pos]
                        val_y_batch = self.label_training[start_pos:end_pos]
                    else:
                        start_pos -= train_size
                        end_pos -= train_size 
                        val_X_batch = self.s2_validation[start_pos:end_pos]
                        val_y_batch = self.label_validation[start_pos:end_pos]

                    yield (val_X_batch, val_y_batch)

                elif channel == 's1_ch5678+s2':
                    if start_pos < train_size:
                        val_X_batch = np.concatenate([self.s1_training[start_pos:end_pos][...,4:], self.s2_training[start_pos:end_pos]], axis=-1)
                        val_y_batch = self.label_training[start_pos:end_pos]
                    else:
                        start_pos -= train_size
                        end_pos -= train_size 
                        val_X_batch = np.concatenate([self.s1_validation[start_pos:end_pos][...,4:], self.s2_validation[start_pos:end_pos]], axis=-1)
                        val_y_batch = self.label_validation[start_pos:end_pos]

                    yield (val_X_batch, val_y_batch)

    #===================================================================
    # Generate round 1 test data for prediction
    #===================================================================
    def getTest1AData(self):
        
        channel = self.data_channel

        if channel == 'full': 
            test1A_data = np.concatenate([self.s1_test1A, self.s2_test1A], axis=-1)
        
        elif channel == 's2_rgb':
            tmp = []
            for p in self.s2_test1A:
                tmp.append(p[...,2::-1])
            test1A_data = np.asarray(tmp)

        elif channel == 's1': 
            test1A_data = self.s1_test1A

        elif channel == 's2': 
            test1A_data = self.s2_test1A

        elif channel == 's1_ch5678': 
            test1A_data = self.s1_test1A[...,4:]

        elif channel == 's1_ch5678+s2':
            test1A_data = np.concatenate([self.s1_test1A[...,4:], self.s2_test1A], axis=-1)
            
        print("Test data shape :", test1A_data.shape)

        return testA_data


    def getTest1BData(self):
        
        channel = self.data_channel

        if channel == 'full': 
            test1B_data = np.concatenate([self.s1_test1B, self.s2_test1B], axis=-1)
        
        elif channel == 's2_rgb':
            tmp = []
            for p in self.s2_test1B:
                tmp.append(p[...,2::-1])
            test1B_data = np.asarray(tmp)

        elif channel == 's1': 
            test1B_data = self.s1_test1B

        elif channel == 's2': 
            test1B_data = self.s2_test1B

        elif channel == 's1_ch5678': 
            test1B_data = self.s1_test1B[...,4:]

        elif channel == 's1_ch5678+s2':
            test1B_data = np.concatenate([self.s1_test1B[...,4:], self.s2_test1B], axis=-1)
            
        print("Test data shape :", test1B_data.shape)

        return test1B_data

    def getTest2AData(self):
        
        channel = self.data_channel

        if channel == 'full': 
            test2A_data = np.concatenate([self.s1_test2A, self.s2_test2A], axis=-1)
        
        elif channel == 's2_rgb':
            tmp = []
            for p in self.s2_test2A:
                tmp.append(p[...,2::-1])
            test2A_data = np.asarray(tmp)

        elif channel == 's1': 
            test2A_data = self.s1_test2A

        elif channel == 's2': 
            test2A_data = self.s2_test2A

        elif channel == 's1_ch5678': 
            test2A_data = self.s1_test2A[...,4:]

        elif channel == 's1_ch5678+s2':
            test2A_data = np.concatenate([self.s1_test2A[...,4:], self.s2_test2A], axis=-1)
            
        print("Test data shape :", test2A_data.shape)

        return test2A_data
