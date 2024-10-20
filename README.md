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

1. Download BDD100K dataset from the [website](http://bdd-data.berkeley.edu/) and Places365 dataset from [here](https://www.kaggle.com/datasets/nickj26/places2-mit-dataset)
2. Download base models for BDD100K experiments from [here](https://drive.google.com/drive/folders/1luVLF8iJ3zUtAghNzSBvi3tAFMFGAWb2?usp=sharing). These models are pretrained on BDD100K dataset.
3. Download retrieved Laion data from [here](https://drive.google.com/drive/folders/1KwBJJ3zXUWUBd6uIImaN2cLngNAfCF0i?usp=drive_link)
4. Download csvs file for datasets splits from [here](https://drive.google.com/drive/folders/1JMxoN7S5zO5iieSSffWiq5wwRK26Vz68?usp=drive_link).
5. Unzip all downloaded files

## 3. Link Downloaded Files
Link downloaded files or move downloaded files to corresponding folders. To generate system links to downloaded files, go to the cloned repo folder, and run below commands.
```
ln -s path_to_folder_for_bdd100k/bdd100k               ./data/datasets/bdd100k
ln -s path_to_folder_for_places365/places365           ./data/datasets/places365
ln -s path_to_folder_for_laion_foggy/laion_foggy       ./data/datasets/laion_foggy
ln -s path_to_folder_for_laion_overcast/laion_overcast ./data/datasets/laion_overcast
ln -s path_to_folder_for_laion_tunnel/laion_tunnel     ./data/datasets/laion_tunnel
ln -s path_to_folder_for_laion_dressing_room/laion_dressing_room   ./data/datasets/laion_dressing_room
ln -s path_to_folder_for_laion_negative/laion_negative   ./data/datasets/laion_negative
ln -s path_to_folder_for_base_models/base_models       ./data/base_models
ln -s path_to_folder_for_csvs/csvs                     ./data/csvs
```



## 4. Run the experiments
Now, we are ready to run experiments. For example,
```
bash train_foggy.sh.
```
We used 2x40G GPUs for BDD100k experiments, which costs around 12 hours for each run.  We used 4x48G GPUs for Places365 experiments, which costs around 44 hours for each run.



Credits: The pipline of our code is based on <https://github.com/locuslab/FLYP>. We thank the authors for making their code publicly available.


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