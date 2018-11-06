# KF 11/06/2018

# AI Challenger competition

import numpy as np
import os
import argparse
import pickle

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset_dir', required=True,
                help='path to input dataset')
ap.add_argument('--data_type', required=True,
				help="data type option: 'train' or 'val'")
ap.add_argument('--output_dir', required=True)
ap.add_argument('--img_size', required=True,
				help='image size option: 224 or 299')
args = vars(ap.parse_args())

data_dir = args['dataset_dir']
data_type = args['data_type']
output_dir = args['output_dir']
img_size = eval(args['img_size'])

img_shape = (img_size, img_size, 3)

data, labels = [], []

## Load labels for both training and validation
if data_type == 'train':
    with open(os.path.join(data_dir, 'AgriculturalDisease_train_annotations.json'), 'r') as f:
	data_dict = json.load(f)
elif data_type == 'val':
    with open(os.path.join(val_dir, 'AgriculturalDisease_validation_annotations.json'), 'r') as f:
	data_dict = json.load(f)
else:
    raise Exception('[KF ERROR] data_type not correct: train or val')

print('')
print('============================================================')
print('                       LOAD DATA') 
print('============================================================')
print('[KF INFO] Total %s sample to be loaded: ' % data_type, len(data_dict))

print("[KF INFO] Loading %s data ..." % data_type)
for item in data_dict:
    image = load_img(os.path.join(data_dir, 'images', item['image_id']), target_size=image_shape)
    image_np = img_to_array(image)
    data.append(image_np)
    labels.append(item['disease_class'])
    print(os.path.join(data_dir, 'images', item['image_id']), " is loaded.")

# Save data and labels into npz file
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_name = data_type + '-' + img_size
print("[KF INFO] Saving %s data and labels into %s.npz ..." % (data_type, save_name))
np.savez_compressed(os.path.join(output_dir, save_name), data=data, labels=labels)
print("[KF INFO] Data and labels are saved successfully!")

