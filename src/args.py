import os
import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help=
        "Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102.",
    )
    
    parser.add_argument(
        "--train-data",
        default=None,
        help="For fine tuning which dataset to train on",
    )
    parser.add_argument(
        "--control-dataset",
        default=None,
        help="For control dataset to train on",
    )
    parser.add_argument(
        "--ct-data-location",
        default='/data/datasets/BDD100k/bdd100k/',
        help="path of control dataset to train on",
    )    

    parser.add_argument(
        "--ct-batch-size",
        type=int,
        default=50,
        help="batch size for constraints",
    )
    
    parser.add_argument(
        "--ct-num-cls-per-batch",
        type=int,
        default=5,
        help="determine how many constraints (or classes) to sample per batch.",
    )
    
    parser.add_argument(
        "--loss",
        default='sog_pnl',
        help="Choose objective loss",
    ) 
    parser.add_argument(
        "--task",
        default='weather',
        help="Choose task type",
    )   
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sampler",
        action=argparse.BooleanOptionalAction,
        help=
        "Choose if use balanced sampler in csv dataset",
    )
    parser.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        help=
        "Choose if lora augmented model",
    )
    parser.add_argument(
        "--lora-dim",
        type=int, 
        default=32,
        )
    parser.add_argument(
        "--ct-sampler",
        default=False,
        action="store_true",
        help=
        "Whether to use balanced sampler for constraints"
    )
    parser.add_argument(
        "--cosine",
        default=False,
        action="store_true",
        help=
        "Whether to use cosine increasing beta"
    )
    parser.add_argument(
        "--pseudo-lable-key",
        default='slabel',
        help="For csv-like datasets, the name of the key for the pseudo-lable.")
    parser.add_argument(
        "--target-class",
        type=int,
        default=1,
        help="For csv-like datasets, the name of the key for the pseudo-lable.")
    
    parser.add_argument("--tau", type=float, default=0.05, help="temperature for ce loss ")
    parser.add_argument("--beta", type=float, default=100, help="hypter-parameter for contraint penalty ")
    parser.add_argument("--t", type=float, default=0.05, help="temperature for sogclr contrastive loss ")
    parser.add_argument("--gamma2", type=float, default=0.8, help="hypter-parameter for moving average for constraints ")
    parser.add_argument("--h-negatives", type=int, default=1, help="number of hard negatives")
    parser.add_argument("--control-size", type=int, default=1000, help="number of samples from each class used for constraints")
    parser.add_argument("--batchs-per-epoch", type=int, default=0, help="number of batchs for one epoch")
    parser.add_argument("--base-model", type=str, default='', help= "The base model we are going to improve")
    parser.add_argument("--num-class", type=int, default=10, help="number of classes")
    ###*************
    
    
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help=
        "Which prompt template is used.",
    )

    
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="batch size for target task",
    )
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")

    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
    )


    
    parser.add_argument("--dataset-type",
                        choices=["webdataset", "csv", "auto"],
                        default="auto",
                        help="Which type of dataset to process.")

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help=
        "Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
                        
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Default random seed.")

    parser.add_argument("--workers",
                        type=int,
                        default=12,
                        help="Number of dataloader workers per GPU.")

    parser.add_argument("--csv-separator",
                        type=str,
                        default=",",##\t
                        help="For csv-like datasets, which separator to use.")
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths.")
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions.")  

    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Repeated run number",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="minimum LR for cosine scheduler",
    )


    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args
