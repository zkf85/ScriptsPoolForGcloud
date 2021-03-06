# 11/07/2018
import numpy as np
import os 
import json
import argparse 
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--model_dir", required=True)
ap.add_argument("-m", "--model_name", required=True,
        help="name of trained model model")
ap.add_argument("-l", "--label_name", required=True,
        help="name of label labels.npz")
ap.add_argument("-i", "--img_dir", required=True,
        help="dir to input image")
ap.add_argument("-o", "--output_name", required=True,
        help="output file name")
ap.add_argument("--img_size", required=True)
args = vars(ap.parse_args())

base_dir = args['model_dir']
model_name = args['model_name']
label_name = args['label_name']
img_dir = args['img_dir']
output_name = args['output_name']
img_size = int(args['img_size'])

img_shape = (img_size, img_size, 3)

# Load label - index dictionary
labels = np.load(os.path.join(base_dir, label_name))
label_dict = labels['class_idx'].tolist()
idx_dict = {y:x for x, y in label_dict.items()}
with open(os.path.join(base_dir, 'idx_to_class.txt'), 'w') as outfile:
    json.dump(idx_dict, outfile, indent=4)

# LOad model
model = load_model(os.path.join(base_dir, model_name))

submission = []
cls_result_dict = {}
counter = 0
for img_name in os.listdir(img_dir):
    print(img_name)
    img = load_img(os.path.join(img_dir, img_name), target_size=img_shape)
    img_np = img_to_array(img)
    img_np = img_np / 255.0
    img_np = np.expand_dims(img_np, axis=0)

    # classify the image and return a index number res_idx
    proba = model.predict(img_np, verbose=1)[0]
    res_idx = np.argmax(proba)
    print('predicted index:', res_idx)
    
    # Get the label number with the result index, then add it and the corresponding file id to the result list.
    tmp_dict = {}
    tmp_dict['disease_class'] = int(idx_dict[res_idx])
    tmp_dict['image_id'] = img_name
    # KF 11/13/2018
    # Save all the results as a python dictionary
    cls_result_dict[img_name] = idx_dict[res_idx]
    submission.append(tmp_dict)
    counter += 1
    print("Image classified: %d" % counter)

# Save to the required json format file
with open(os.path.join(base_dir, output_name), 'w') as outfile:
    json.dump(submission, outfile, indent=4)
with open(os.path.join(base_dir, 'result_dict.pkl'), 'wb') as f:
    pickle.dump(cls_result_dict, f)
