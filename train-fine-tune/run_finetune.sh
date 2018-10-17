# Updated KF 10/17/2018
# KF 10/12/2018

date="10122018"

npz_file_dir=~/disk/datasets/dataset224.npz
img_size=224
pretrained="MobileNetV2"
epochs=50
batch_size=64
learning_rate=1e-3

base_dir=~/disk/leaves-classification
model_name=model-$pretrained-$img_size-$epochs-$batch_size-$learning_rate-$date
mkdir $base_dir/$model_name

# Training
exit_code=python training.py \
--directory $base_dir \
--dataset $npz_file_dir \
--pretrained $pretrained \
--img_size $img_size \
--epochs $epochs \
--batch_size $batch_size \
--learning_rate $learning_rate \
--model leaves.model \
--labelbin labelbin.pkl \
--aug

# Upload the package to my bucket and shutdown
if [ exit_code -eq 0 ]
then
	gsutil -m cp -r . gs://kf-bucket/$model_name
	gcloud compute instances stop --zone=us-east1-b kf-gpu
fi
