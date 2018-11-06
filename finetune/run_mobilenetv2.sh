# Updated KF 10/17/2018
# KF 10/12/2018

date="10172018"

#npz_file_dir=~/disk/datasets/4-cls-leaves/dataset224.npz
npz_file_dir=~/kf-bucket/dataset224.npz
img_size=224
pretrained="MobileNetV2"
epochs=20
batch_size=64
learning_rate=1e-4

save_dir=~/disk/results
model_name=leaves-3-cls-$pretrained-$img_size-$epochs-$batch_size-$learning_rate-$date
#mkdir $save_dir/$model_name

# Training
python3 02_train_with_npz.py \
--directory $save_dir \
--dataset $npz_file_dir \
--pretrained $pretrained \
--img_size $img_size \
--epochs $epochs \
--batch_size $batch_size \
--learning_rate $learning_rate \
--model leaves.model \
--labelbin labelbin.pkl \
--test \
--aug \
#> $save_dir/$model_name/log.txt

# Upload the package to my bucket and shutdown
if [ $? -eq 0 ]
then
	gsutil -m cp -r $save_dir gs://kf-bucket/
	gcloud compute instances stop --zone=us-east1-b kf-gpu
fi
