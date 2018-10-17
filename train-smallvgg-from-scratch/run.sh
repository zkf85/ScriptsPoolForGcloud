# KF 10/12/2018

root_dir=..
npz_file_dir=$root_dir/dataset224.npz

date="10122018"
model_name=leaves-model-smallvgg-$date

# Training
python smallvgg_training.py \
--dataset $npz_file_dir \
--model leaves.model \
--labelbin labelbin.pkl \
--aug

# Upload the package to my bucket and shutdown
autorun=1
if [ $autorun = 1 ]; then
	tar -zcvf $model_name.tar.gz .
	gsutil -m cp $model_name.tar.gz gs://kf-bucket
	gcloud compute instances stop --zone=us-east1-b kf-gpu
fi
