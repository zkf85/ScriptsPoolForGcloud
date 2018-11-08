
model_file=ckpt-weight-best.hdf5
test_img_dir=/home/kefeng/disease_datasets/dataset/AgriculturalDisease_testA/images
output_name=kf_submission_01.json


python3 03_classify.py \
-d . \
-m $model_file \
-l labels.npz \
-i $test_img_dir \
-o $output_name \
--img_size 224
