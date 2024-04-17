if [ ! -d "/home/yc2367/BitSim/sim/model_profile/meters/sparten_saved_dict" ]; then
    mkdir /home/yc2367/BitSim/sim/model_profile/meters/sparten_saved_dict
fi

batch_size=128

for model in vgg16 resnet34 resnet50 vit\-small vit\-base bert\-mrpc bert\-sst2;
do
    python3 -W ignore ./test_sparten.py \
        --model ${model} \
        --batch_size ${batch_size} \
        --train_dir "/share/abdelfattah/imagenet/train/" \
        --val_dir "/share/abdelfattah/imagenet/val/" \
        --sparten_save_dir "/home/yc2367/BitSim/sim/model_profile/meters/sparten_saved_dict"
done
