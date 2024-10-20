import os
import torch
from tqdm import tqdm

import numpy as np

import clip.clip as clip

import src.templates as templates
import src.datasets as datasets

from src.args import parse_arguments
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.eval import evaluate



def get_zeroshot_classifier(args, clip_model):
    # assert args.template is not None
    assert args.control_dataset is not None
    # # template = args.control_dataset.template # getattr(templates, args.template) 
    # # logit_scale = clip_model.logit_scale
    
    dataset_class = getattr(datasets, args.control_dataset)
    dataset = dataset_class(None,
                            location=args.ct_data_location,
                            task=args.task,
                            mode=None,
                            )

    device = args.device
    clip_model.eval()
    clip_model.to(device)

    template = dataset.template
    with torch.no_grad():
        zeroshot_weights = []
        for cls_id, classname in enumerate(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device)  # tokenize
            if args.lora:
                embeddings = clip_model.encode_text(
                    texts, cls_ids = torch.tensor([cls_id]))  # embed with text encoder   #[80,512] [num_templates, 512] #[7,512]
            else:
                embeddings = clip_model.encode_text(
                    texts)  # embed with text encoder                
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm() #[1, 512]

            zeroshot_weights.append(embeddings) 

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device) #[1000,1,512]
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2) #[512,1,1000]

        zeroshot_weights = zeroshot_weights.squeeze().float() #[512,1000]
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1) #[1000,512] weather[6,512]

    classification_head = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights)
    
    classification_head_params = list(classification_head.parameters())


    return classification_head


def eval(args):
    args.freeze_encoder = True
    if args.load is not None:
        classifier = ImageClassifier.load(args.load)
    else:
        #image_encoder = ImageEncoder(args, keep_lang=True)
        image_encoder = CLIPEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args,
                                                      image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder,
                                     classification_head,
                                     process_images=False)

    evaluate(classifier, args)

    if args.save is not None:
        classifier.save(args.save)


if __name__ == '__main__':
    args = parse_arguments()
    eval(args)