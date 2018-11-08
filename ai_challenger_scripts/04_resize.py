# KF 10/14/2018

# Resize all the images and save them in to a 'small' subfolder

import os
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--directory', required=True,
                help='image directory to be resized.')
ap.add_argument('-o', '--output_dir', required=True,
                help='output directory as the same folder tree as the original image directory')
ap.add_argument('-i', '--img_size', type=int, default=299)
args = vars(ap.parse_args())

img_dir = args['directory']
small_dir = args['output_dir']
img_size = args['img_size']

dim = (img_size, img_size)
# Check if the subfolder exists, if not ,create one
if not os.path.exists(small_dir):
    os.makedirs(small_dir)


for sub_dir in os.listdir(img_dir):
    if not os.path.exists(os.path.join(small_dir, sub_dir)):
        os.makedirs(os.path.join(small_dir, sub_dir))
    i = 0
    for subsub_dir in os.listdir(os.path.join(img_dir, sub_dir)):
        if not os.path.exists(os.path.join(small_dir, sub_dir, subsub_dir)):
            os.makedirs(os.path.join(small_dir, sub_dir, subsub_dir))

        file_names = sorted([f for f in os.listdir(os.path.join(img_dir, sub_dir, subsub_dir)) if f.lower().endswith('.jpg')])

        # Resizing and save into subfolder
        for name in file_names:
            img = cv2.imread(os.path.join(img_dir, sub_dir, subsub_dir, name))

            img_small = cv2.resize(img, dim)
            img_small_path = os.path.join(small_dir, sub_dir, subsub_dir, name)
            status = cv2.imwrite(img_small_path, img_small)
            #if status:
            #    print('%s resized copied to %s' % (name, img_small_path)) 
            i += 1
        print(i, 'resized in', os.path.join(small_dir, sub_dir, subsub_dir))
print('[KF INFO] Resizing complete!')
