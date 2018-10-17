# KF 10112018

# Preprocess the data for later training, save the processed data as binary for later use
# The saved binary dataset includes both images and corresponding labels.
import numpy as np
import os
import random
import argparse
import pickle

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')

args = vars(ap.parse_args())                

image_dim = (224, 224, 3)


# Get dataset directory
data_dir = args['dataset']

# Small size or not
small = False
small_size = 10

# Initialize data and labels
data, labels = [], []

# Grab image paths and randomly shuffle them
print('[KF INFO] Loading all images in all subfolders ...')
categories = [item for item in os.listdir(data_dir) if not item.startswith('.')]
cls_number = len(categories)
print(categories)
print('[KF INFO] Total class number :', cls_number)

imgPaths = []
img_numbers = []

for c in categories:
    imgPath = [os.path.join(data_dir, c, name) for name in os.listdir(os.path.join(data_dir, c)) if name.lower().endswith(('.jpg', '.png'))]
    if small:
        imgPath = imgPath[:small_size]
    print('[KF INFO] Number of images in category %s is %d' % (c, len(imgPath)))
    imgPaths += imgPath

# Shuffle the image paths
random.seed(10)
random.shuffle(imgPaths)

# Load images
count = 0
for imgPath in imgPaths:
    image = load_img(imgPath, target_size=image_dim)
    image_np = img_to_array(image)
    data.append(image_np)

    # Extract label
    label = imgPath.split(os.path.sep)[-2]
    #print(imgPath, label) 
    labels.append(label)

print('[KF INFO] %d images in total are loaded.' % len(data))

np.savez_compressed('dataset', data=data, labels=labels, cls_number=cls_number)
print('[KF INFO] data and labels are saved successfully!')
