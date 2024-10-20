import os
import json

import torch
import numpy as np
from src.train_eval import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.datasets as datasets
import torch.nn.functional as F


def eval_single_dataset(image_classifier, dataset, args, train_stats, classification_head):

    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    classification_head.eval()

    dataloader = get_dataloader(dataset,
                                is_train=False,
                                args=args,
                                image_encoder=image_enc)

    batched_data = enumerate(dataloader)
    device = args.device


    classnames = dataset.classnames 
    all_labels, all_logits = [], []
        
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        class_correct = [0] *  len(classnames) 
        class_total = [0] * len(classnames)
        # labels = [i for i in range( len(dataset.classnames) )]
        for i, data in batched_data:

            data = maybe_dictionarize(data)
            x = data[input_key].to(device) #[5,3,224,224]
            y = data['labels'].to(device) #[5] labels eg:[502,502,502,502,502] weather[0,0,0,0,0]

            if 'image_paths' in data:
                image_paths = data['image_paths']
            
            logits = utils.get_logits(x, model, classification_head) ##use classification head to make prediction
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            preds = logits.argmax(dim=1, keepdim=True)#.to(device) #[batch_size, 1] 
            
            # #####################
            correct += preds.eq(y.view_as(preds)).sum().item()
            n += y.size(0)

            # print('label',label,'y',y)
            y = y.detach().cpu().reshape(-1)
            preds = preds.detach().cpu().reshape(-1)
            for label in y:
                class_total[label] = class_total[label] + 1
            for label, pred in zip(y, preds):
                if label == pred:
                    class_correct[label] = class_correct[label] + 1

            all_labels.append(y)
            all_logits.append(logits.cpu().clone().detach())

        top1 = correct / n
        #### recoding loss values
        # if dataset.mode in ['train', 'val']:
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)
        losses = F.cross_entropy(all_logits/args.tau, all_labels, reduction='none')


    #### recording accuracy
    train_stats[dataset.mode+args.task] = round(top1, 4)

    for i, classname in enumerate(classnames):
        ### acc for each class
        train_stats[dataset.mode+classname] = round(class_correct[i]/class_total[i], 4)
        if dataset.mode != 'test':
            train_stats[dataset.mode+classname+'_loss'] = (losses[all_labels==i]).mean().item()
    ### 
    train_stats[dataset.mode+'_ave'] = round(np.mean(
        [(class_correct[i]/class_total[i]) for i in range(len(classnames))]),   4)




def evaluate(image_classifier,
             args,
             classification_head,
             train_stats={},
             logger=None,):
    if args.eval_datasets is None:
        return
    
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(image_classifier.module.val_preprocess,
                                control_size = args.control_size,
                                location=args.ct_data_location,
                                batch_size=max(args.batch_size, args.ct_batch_size),
                                num_workers=args.workers,
                                task=args.task
                                )

        eval_single_dataset(image_classifier, dataset, args, train_stats,
                                      classification_head)

