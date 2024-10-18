# Model Developmental Safety: A Safety-Centric Method and Applications in Vision-Language Models

This repo is to provide implementation of the algorithm proposed in paper [Model Developmental Safety: A Safety-Centric Method and Applications in Vision-Language Models](https://arxiv.org/abs/2410.03955) and reproduce the experimental results.


## 1. Setup Environment
* Git clone this repo
```
git clone https://github.com/GangLii/DevSafety
```
* Create conda environment
```
conda create -n DevSafety python=3.10
conda activate DevSafety
cd ./DevSafety
pip install -r requirements.txt
```

## 2. Datasets and Base Models

1. Download BDD100K dataset from the [website]() and Places365 dataset from [here]()
2. Download base models for BDD100K experiments from [here](). These models are pretrained on BDD100K dataset.
3. Download retrieved Laion data from [here]()
4. Download csvs file from [here](), which contaions data sets splits.


## 3. Move Downloaded Files
Move downloaded files to coresponding folders. Specifically,
* csv.zip -> ./dataset/csvs

## 4. Run the experiments
Now, we are ready to run experiments. For example,
```
bash train_foggy.sh.
```
We used 2x40G GPUs for BDD100k experiments, which costs around 12 hours for each run.  We used 4x48G GPUs for Places365 experiments, which costs around 44 hours for each run.



Credits: The pipline of our code is based on <https://github.com/locuslab/FLYP>. We thank the authors for open sourcing their code.


## Contact
If you have any questions, please open a new issue or contact Gang Li via <gang-li@tamu.edu>. If you find this repo helpful, please cite the following paper:
```
@article{li2024model,
  title={Model Developmental Safety: A Safety-Centric Method and Applications in Vision-Language Models},
  author={Li, Gang and Yu, Wendi and Yao, Yao and Tong, Wei and Liang, Yingbin and Lin, Qihang and Yang, Tianbao},
  journal={arXiv preprint arXiv:2410.03955},
  year={2024}
}
```