if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=vgg16_bn
epochs=200
batch_size=64
lr=0.1
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="inference.log"
wbit=8
abit=8
xqtype="lsq"
wqtype="minmax_channel"
ttype=ptq

# Column Pruning
flag=0
grp_size=16
N=2
pr=0.95

save_path="/home/jm2787/BitSim/save/${model}/ptq/lsq_minmax_channel/${model}_w8_a8_lr1e-3_batch128_cross_entropyloss/structured_columnprune${flag}/grp${grp_size}/column${N}/pr${pr}/"
# save_path="/home/jm2787/BitSim/save/${model}/ptq/lsq_minmax_channel/${model}_w8_a8_lr1e-3_batch128_cross_entropyloss/baseline/"
pre_trained="/home/jm2787/BitSim/save/vgg16_bn/ptq/lsq_minmax_channel/vgg16_bn_w8_a8_lr1e-3_batch128_cross_entropyloss/model_best.pth.tar"
# pre_trained="/home/jm2787/BitSim/save/resnet50/ptq/lsq_minmax_channel/resnet50_w8_a8_lr1e-3_batch128_cross_entropyloss/model_best.pth.tar"
# pre_trained="/home/jm2787/BitSim/save/resnet34/ptq/lsq_minmax_channel/resnet34_w8_a8_lr1e-3_batch128_cross_entropyloss/model_best.pth.tar"

python3 -W ignore ./imagenet/t2c.py \
    --save_path ${save_path} \
    --log_file ${log_file} \
    --batch_size ${batch_size} \
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
    --flag ${flag} \
    --grp_size ${grp_size} \
    --N ${N} \
    --prune_ratio ${pr} \
