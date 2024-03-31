if [ ! -d "/home/yc2367/BitSim/sim/model_profile/meters/sparten_saved_dict" ]; then
    mkdir /home/yc2367/BitSim/sim/model_profile/meters/sparten_saved_dict
fi

model=resnet50
batch_size=128

python3 -W ignore ./test_sparten.py \
    --model ${model} \
    --batch_size ${batch_size} \
    --train_dir "/share/abdelfattah/imagenet/train/" \
    --val_dir "/share/abdelfattah/imagenet/val/" \
    --sparten_save_dir "/home/yc2367/BitSim/sim/model_profile/meters/sparten_saved_dict"
