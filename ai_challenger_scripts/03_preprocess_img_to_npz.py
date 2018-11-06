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
				help="data type option: 'train' or 'va'")
ap.add_argument('--output_dir', required=True)
ap.add_argument('--img_size', required=True,
				help='image size option: 224 or 299')
args = vars(ap.parse_args())

img_size 
