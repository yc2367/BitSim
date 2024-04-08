if [ -d "./plot" ]; then
  rm -rf ./plot
fi;

mkdir plot
# python resnet18_bit_prune_plot.py
python bit_prune_plot.py --model resnet50