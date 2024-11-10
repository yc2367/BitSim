if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=vit_small
epochs=200
batch_size=128
lr=0.1
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="inference.log"
wbit=8
abit=8
xqtype="minmax_token"
wqtype="minmax_channel"
ttype=ptq

# Column Pruning
flag=2
grp_size=32
N=4

save_path="./save/imagenet/${model}/${xqtype}_${wqtype}/imagenet/${model}/${model}_w8_a8_lr1e-4_batch100_cross_entropyloss_all/columnprune${flag}/grp${grp_size}/structured_column${N}/"
pre_trained="/home/jm2787/MLSys24/T2C/save/imagenet/vit_small/minmax_token_minmax_channel/vit_small_w8_a8_lr1e-4_batch100_cross_entropyloss_all/model_best.pth.tar"

python3 -W ignore ./imagenet/t2c.py \
    --save_path ${save_path} \
    --log_file ${log_file} \
    --model ${model} \
    --resume ${pre_trained} \
    --fine_tune \
    --wqtype ${wqtype} \
    --xqtype ${xqtype} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --train_dir "/share/seo/imagenet/train/" \
    --val_dir "/share/seo/imagenet/val/" \
    --evaluate \
    --trainer qattn \
    --swl 32 \
    --sfl 26 \
    --export_samples 8 \
    --flag ${flag} \
    --grp_size ${grp_size} \
    --N ${N} \
    
