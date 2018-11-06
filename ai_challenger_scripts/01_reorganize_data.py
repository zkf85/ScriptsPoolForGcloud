# KF 11/05/2018
# Make a copy of the dataset in the order keras flow_from_directory needs

import os
import argparse
import json
import shutil

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset_dir', required=True,
				help='path to the training dataset directory')
ap.add_argument('-o', '--output_dir', required=True,
				help='path to the output dir')
args = vars(ap.parse_args())

data_dir = args['dataset_dir']
output_dir = args['output_dir']

if not os.path.exists(output_dir):
	os.makedirs(output_dir)
for sub in ['train', 'val', 'testA']:
	sub_dir = os.path.join(output_dir, sub)
	if not os.path.exists(sub_dir):
		os.makedirs(sub_dir)

	
	for i in range(61):
		cls_dir = os.path.join(sub_dir, str(i))
		if not os.path.exists(cls_dir):
			os.makedirs(cls_dir)

# Copy
original = os.path.join(data_dir, 'AgriculturalDisease_validationset')
with open(os.path.join(original, 'AgriculturalDisease_validation_annotations.json'), 'r') as f:
	json_list = json.load(f)
	print(len(json_list))

i = 0
for item in json_list:
	file_path = os.path.join(original, 'images', item['image_id'])
	shutil.copyfile(file_path, os.path.join(output_dir, 'val', str(item['disease_class']), item['image_id']))
	i += 1

print(i)



