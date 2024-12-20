#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD" 

for seed in 0 
do
  echo "Iteration seed $seed "
  for beta in 100 
  do 
    echo "Iteration beta $beta "
    CUDA_VISIBLE_DEVICES=0,1 python src/main.py  \
        --train-data="./data/csvs/overcast/integrated_laion_bddovercast_cons2k_neg10t.csv" \
        --ct-data-location="./data/datasets/bdd100k"     \
        --base-model="./data/base_models/checkpoint_weather.pt"   \
        --control-dataset=BDD100k --control-size=2000 --epochs=40 --lr=1e-6 --wd=0.1 --workers=8   \
        --batch-size=256 --ct-batch-size=40 --beta=$beta  --tau=0.05 --t=0.05 \
        --model=ViT-B/16  --csv-img-key=filepath --csv-caption-key=title   --pseudo-lable-key=slabel  \
        --sampler --loss=sog_pnl --seed=$seed --gamma=0.8  --batchs-per-epoch=600  --lora-dim=32  \
        --task='weather_enhance' --ct-sampler --ct-num-cls-per-batch=4  --target-class=1 --lora --num-class=10 \
        --eval-datasets=BDD100k,BDD100k_Val,BDD100k_Test  --save=./checkpoints/    \
        --exp_name=bdd100k_overcast/sogpnl_subsetD2k_cln_bddV3_E40B600lr6beta${beta}tau0005t005
  done    
done