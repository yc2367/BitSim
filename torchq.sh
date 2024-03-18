if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=resnet18
epochs=200
batch_size=128
lr=0.1
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="training.log"
wbit=8
abit=8
xqtype="lsq"
wqtype="adaround"
ttype=ptq

save_path="/home/jm2787/BitSim/save/resnet18_torchvision/t2c/"

python3 -W ignore ./torchq.py \
    --save_path ${save_path} \
    --model ${model} \
    --wqtype ${wqtype} \
    --xqtype ${xqtype} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --train_dir "/share/seo/imagenet/train/" \
    --val_dir "/share/seo/imagenet/val/" \
    --evaluate \
    --trainer ptq \
    --flag 0 \
    --grp_size 32 \
    --N 5 \
    --hamming_distance 0.5 \
