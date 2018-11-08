# 11/07/2018
import numpy as np
import os 
import json
import argparse 
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model_path", required=True,
        help="path to trained model model")
ap.add_argument("-l", "--label_path", required=True,
        help="path to label labels.npz")
ap.add_argument("-i", "--img_dir", required=True,
        help="dir to input image")
ap.add_argument("--img_size", required=True)
args = vars(ap.parse_args())

model_path = args['model_path']
img_size = args['img_size']
img_dir = args['img_dir']

img_shape = (img_size, img_size, 3)

# Load label - index dictionary
labels = np.load(args['label_path'])
label_dict = labels['class_idx'].tolist()
idx_dict = {y:x for x, y in in label_dict.items()}

# LOad model
model = load_model(model_path)

submission = []
for img_name in os.listdirs(img_dir):
    img = load_img(os.path.join(img_dir, img_name), target_size=img_shape)
    img_np = img_to_array(img)
    img_np = img_np / 255.0
    img_np = np.expand_dims(img_np, axis=0)

    # classify the image and return a index number res_idx
    proba = model.predict(img_np, verbose=1)[0]
    res_idx = np.argmax(proba)
    
    # Get the label number with the result index, then add it and the corresponding file id to the result list.
    tmp_dict = {}
    tmp_dict['image_id']: img_name
    tmp_dict['disease_class']: int(idx_dict[res_idx])
    submission.append(tmp_dict)

# Save to the required json format file
with open('kf_submission.json', 'w') as outfile:
    json.dump(submission, outfile, indent=4)


