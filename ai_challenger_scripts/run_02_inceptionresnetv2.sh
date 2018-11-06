# Updated KF 10/17/2018
# KF 10/12/2018

date="11062018"

dataset_dir=~/disk/disease_datasets/dataset_for_keras
pretrained="InceptionResNetV2"
img_size=299
epochs=20
batch_size=64
learning_rate=2e-4

model_name=aichallenger-disease-$date-$pretrained-$img_size-$epochs-$batch_size-$learning_rate
save_dir=~/disk/results/$model_name
#mkdir -p $save_dir

# Training
python3 02_train.py \
--dataset $dataset_dir \
--save_dir $save_dir \
--pretrained $pretrained \
--img_size $img_size \
--epochs $epochs \
--batch_size $batch_size \
--learning_rate $learning_rate \
--model disease.model \
#> $save_dir/log.txt
#
## Upload the package to my bucket and shutdown
#if [ $? -eq 0 ]
#then
#	gsutil -m cp -r $save_dir gs://kf-bucket/
#fi
#gcloud compute instances stop --zone=us-east1-b kf-gpu
