data_dir=~/disk/disease_datasets/dataset/AgriculturalDisease_trainingset
data_type=train
img_size=299
output_dir=~/disk/disease_datasets/dataset_npz

python3 03_preprocessing_img_to_npz.py \
--dataset_dir $data_dir
--data_type $data_type
--img_size $img_size
--output_dir $output_dir

