#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD" 


for seed in 0
do
  echo "Iteration seed $seed "
  for beta in 1000
  do
    echo "Iteration beta $beta "
    python src/main.py  \
        --train-data="./data/csvs/dressing_room/integrated_laion_dressing_cons2k_neg10t.csv" \
        --ct-data-location="./data/datasets/places365"     \
        --base-model=""    \
        --control-dataset=Places365 --control-size=2000 --epochs=40 --lr=1e-6 --wd=0.1  --workers=24  \
        --batch-size=64 --ct-batch-size=480 --beta=$beta  --tau=0.01 --t=0.05   \
        --model=ViT-B/16  --csv-img-key=filepath --csv-caption-key=title   --pseudo-lable-key=slabel  \
        --sampler --loss=sog_pnl --seed=$seed --gamma2=0.8  --batchs-per-epoch=1600  --lora-dim=32  \
        --task='places365' --ct-sampler --ct-num-cls-per-batch=240  --target-class=1 --lora --num-class=370   \
        --eval-datasets=Places365,Places365_Val,Places365_Test  --save=./checkpoints/   \
        --exp_name=places365_dressing/sogpnl_2k_ct240_E40B1600lr6beta${beta}tau001t005
  done
done

