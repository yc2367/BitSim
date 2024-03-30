if [ ! -d "./save" ]; then
    mkdir ./save
fi

model=resnet50
batch_size=64

python3 -W ignore ./test_sparten.py \
    --model ${model} \
    --batch_size ${batch_size} \
    --train_dir "/share/abdelfattah/imagenet/train/" \
    --val_dir "/share/abdelfattah/imagenet/val/" \
