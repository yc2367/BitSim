if [ -d "./plot" ]; then
  rm -rf ./plot
fi;

mkdir plot
python resnet50_bit_prune_plot.py