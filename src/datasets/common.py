import os
import torch
import json
import glob
import collections
import random

import numpy as np

from tqdm import tqdm

import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler

class ClassSampler(Sampler):
    def __init__(self, labels, batch_size, num_cls_per_batch = 5):

        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.label_dict, self.pos_ptr = {}, {}
        for cls in self.classes:
            self.label_dict[cls] = (self.labels == cls).nonzero()[0]
            self.pos_ptr[cls] = 0
        ####To avoid sampling error
        while len(self.classes) < num_cls_per_batch: 
            self.classes = np.concatenate((self.classes, self.classes))
            
        self.cls_ptr, self.classes = 0, np.random.permutation(self.classes)
        self.num_cls = num_cls_per_batch
        self.batch_size = batch_size

        ### we assume that batch_size is divisible by num_cls_per_batch 
        assert self.batch_size % num_cls_per_batch == 0   
        self.num_pos = self.batch_size // self.num_cls 

        #### to define an epoch
        # ### for DP
        self.num_batch = len(labels)//self.batch_size #//2 
        self.ret = np.empty(self.num_batch*self.batch_size, dtype=np.int64)


    def __iter__(self):

        for batch_id in range(self.num_batch):
            if self.cls_ptr+self.num_cls >= len(self.classes):
                temp = self.classes[self.cls_ptr:]
                np.random.shuffle(self.classes)
                self.cls_ptr = (self.cls_ptr+self.num_cls)% len(self.classes)
                # print(self.classes, self.cls_ptr)
                cls_ids = np.concatenate((temp, self.classes[:self.cls_ptr] ))
            else:
                cls_ids = self.classes[self.cls_ptr:self.cls_ptr+self.num_cls]
                self.cls_ptr = self.cls_ptr + self.num_cls
                
            beg = batch_id*self.batch_size
            for cls_id in cls_ids:
                if self.pos_ptr[cls_id]+self.num_pos >= len(self.label_dict[cls_id]):
                    temp = self.label_dict[cls_id][self.pos_ptr[cls_id]:]
                    np.random.shuffle(self.label_dict[cls_id])
                    self.pos_ptr[cls_id] = (self.pos_ptr[cls_id]+self.num_pos)%len(self.label_dict[cls_id])
                    self.ret[beg:beg+self.num_pos]= np.concatenate((temp,self.label_dict[cls_id][:self.pos_ptr[cls_id]]))
                else:
                    self.ret[beg:beg+self.num_pos]= self.label_dict[cls_id][self.pos_ptr[cls_id]: self.pos_ptr[cls_id]+self.num_pos]
                    self.pos_ptr[cls_id] += self.num_pos
                beg += self.num_pos

        # subsample
        # indices = self.ret[self.rank:len(self.ret):self.num_replicas]
        return iter(self.ret) 


    def __len__ (self):
        return len(self.ret)

class DataSampler(Sampler):
    def __init__(self, labels, batch_size, pos_num=1):
        r"""Arguments:
            labels (list or numpy.array): labels of training dataset, the shape of labels should be (n,)
            batch_size (int): how many samples per batch to load
            pos_num (int): specify how many positive samples in each batch
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_num, self.neg_num = pos_num, self.batch_size - pos_num

        self.posList = np.flatnonzero(self.labels==1)
        self.negList = np.flatnonzero(self.labels==0)
        self.posLen, self.negLen = len(self.posList), len(self.negList)
        self.posPtr, self.negPtr = 0, 0

        np.random.shuffle(self.posList)
        np.random.shuffle(self.negList)

        #### define how many batchs per epoch
        # self.batchNum = max(self.posLen//self.pos_num, self.negLen//self.neg_num)
        self.batchNum = len(self.labels)//self.batch_size
        self.ret = np.empty(self.batchNum*self.batch_size, dtype=np.int64)

    def __iter__(self):
        r""" This functuin will return a new Iterator object for an epoch."""

        for batch_id in range(self.batchNum):
            #### load postive samples
            beg = batch_id*self.batch_size
            if self.posPtr+self.pos_num >= self.posLen:
                temp = self.posList[self.posPtr:]
                np.random.shuffle(self.posList)
                self.posPtr = (self.posPtr+self.pos_num)%self.posLen
                self.ret[beg:beg+self.pos_num]= np.concatenate((temp,self.posList[:self.posPtr]))
            else:
                self.ret[beg:beg+self.pos_num]= self.posList[self.posPtr: self.posPtr+self.pos_num]
                self.posPtr += self.pos_num

            ### load negative samples
            beg += self.pos_num
            if self.negPtr+self.neg_num >= self.negLen:
                temp = self.negList[self.negPtr:]
                np.random.shuffle(self.negList)
                self.negPtr = (self.negPtr+self.neg_num)%self.negLen
                self.ret[beg:beg+self.neg_num]= np.concatenate((temp,self.negList[:self.negPtr]))
            else:
                self.ret[beg:beg+self.neg_num]= self.negList[self.negPtr: self.negPtr+self.neg_num]
                self.negPtr += self.neg_num

        return iter(self.ret)


    def __len__ (self):
        return len(self.ret)
    
class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f'Flipping labels with probability {self.flip_label_prob}')
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes-1)
                    self.samples[i] = (
                        self.samples[i][0],
                        new_label
                    )

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch


def get_features_helper(image_encoder, dataloader, device, noscale):
    all_data = collections.defaultdict(list)
    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(image_encoder, device_ids=[x for x in range(torch.cuda.device_count())])
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            image_encoder = image_encoder.to(inputs.device)
            features = image_encoder(inputs)
            # if noscale:
            #     features = features / features.norm(dim=-1, keepdim=True)
            # else:
            #     logit_scale = image_encoder.module.model.logit_scale
            #     features = logit_scale.exp() * features

            all_data['features'].append(features.cpu())

            for key, val in batch.items():
                if key == 'images':
                    continue
                if hasattr(val, 'cpu'):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device, cache_dir, noscale):
    split = 'train' if is_train else 'val'
    dname = type(dataset).__name__
    # import pdb;pdb.set_trace()
    if cache_dir is not None:
        cache_dir = f'{cache_dir}/{dname}/{split}'
        cached_files = glob.glob(f'{cache_dir}/*')
    if cache_dir is not None and len(cached_files) > 0:
        print(f'Getting features from {cache_dir}')
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        loader = dataset.train_loader if is_train else dataset.test_loader
        data = get_features_helper(image_encoder, loader, device, noscale)
        if cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Caching data at {cache_dir}')
            for name, val in data.items():
                torch.save(val, f'{cache_dir}/{name}.pt')
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device, cache_dir=None, noscale=True):
        self.data = get_features(is_train, image_encoder, dataset, device, cache_dir, noscale)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data


def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device, args.cache_dir, args.noscale)
        dataloader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=is_train)
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader