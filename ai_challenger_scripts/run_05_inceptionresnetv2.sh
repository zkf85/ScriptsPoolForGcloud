# run_05_train_concat
#
# Updated - KF 11/09/2018
#  added the trainset_option parameter to control the input training data folder option
# Updated - KF 11/08/2018
# Updated - KF 10/17/2018
# KF 10/12/2018

####################################################################################################
# Parameters 
####################################################################################################

# --------------------------------------------------------------------------------------------------
# Set date
# --------------------------------------------------------------------------------------------------
# REMEMBER TO CHANGE THE DATE FIRST!!!!!!
date="11092018"

# --------------------------------------------------------------------------------------------------
# Dataset base directory path
# --------------------------------------------------------------------------------------------------
#dataset_dir=~/disk/disease_datasets/dataset_for_keras
dataset_dir="/home/kefeng/disease_datasets/299_dataset_for_keras"

# --------------------------------------------------------------------------------------------------
# Training mode:
# --------------------------------------------------------------------------------------------------
training_mode="trial"

# --------------------------------------------------------------------------------------------------
# Basic training parameters
# --------------------------------------------------------------------------------------------------
pretrained="InceptionResNetV2"
img_size=299
epochs=20
batch_size=64

# --------------------------------------------------------------------------------------------------
# Training dataset option: either 'concat' or 'train'
# --------------------------------------------------------------------------------------------------
trainset_option="concat"
#trainset_option=train

# --------------------------------------------------------------------------------------------------
# Optimizer option
# --------------------------------------------------------------------------------------------------
optimizer="nadam"
#optimizer=rmsprop

# --------------------------------------------------------------------------------------------------
# Model saving setting
# --------------------------------------------------------------------------------------------------
model_dir_name="aichallenger-disease-concat-$date-$pretrained-$img_size-$epochs-$batch_size-$optimizer-$trainset_option"
#save_dir=~/disk/results/$model_name
save_dir="/home/kefeng/results/$model_dir_name"
mkdir -p $save_dir
#cp 03_classify.py $save_dir
#cp run_03.sh $save_dir
model_file_name="disease.model"

####################################################################################################
# Run the script with parameters
####################################################################################################

# Training
python3 05_train_concat.py \
--training_mode     $training_mode \
--dataset           $dataset_dir \
--save_dir          $save_dir \
--model_file_name   $model_file_name \
--training_mode     $training_mode \
--pretrained        $pretrained \
--img_size          $img_size \
--epochs            $epochs \
--batch_size        $batch_size \
--trainset_option   $trainset_option \
--optimizer         $optimizer \
#> $save_dir/log.txt

## Upload the package to my bucket and shutdown
#if [ $? -eq 0 ]
#then
#	gsutil -m cp -r $save_dir gs://kf-bucket/
#fi
#gcloud compute instances stop --zone=us-east1-b kf-gpu
