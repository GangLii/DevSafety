import os
import torch

from .common import ImageFolderWithPaths, SubsetSampler, ClassSampler
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchvision
import numpy as np
from PIL import Image
import glob
from .bdd100k_classnames import get_classnames
import src.templates as templates


class BDD100k_Dataset(Dataset):
    def __init__(self, root, image_list, labels, transform=None):
        self.root_dir = root
        self.transforms = transform
        self.img_list = image_list
        self.labels = labels


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        try:
            path = self.root_dir + self.img_list[idx]
            image = Image.open(path)
        except:
            image = Image.open(self.img_list[idx])
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[idx]
        
        return image, label, idx
    


class BDD100k:
    def __init__(
        self,
        preprocess,
        preprocess_eval=None, ### for reference evaluation
        control_size = 1000,
        location= '/data/datasets/BDD100k/bdd100k/',
        batch_size=128,
        ct_batch_size = 50,
        ct_num_cls_per_batch = 5,
        ct_sampler = False,
        num_workers=12,
        task='weather',
        mode='train',  ## val , test
    ):
        self.preprocess = preprocess
        if preprocess_eval is not None:
            self.preprocess_eval = preprocess_eval 
        else:
            self.preprocess_eval = preprocess
        self.location = location
        self.batch_size = batch_size
        self.ct_batch_size = ct_batch_size
        self.ct_num_cls_per_batch = ct_num_cls_per_batch
        self.ct_sampler = ct_sampler
        self.num_workers = num_workers
        self.classnames = get_classnames(task)
        self.control_classnames = get_classnames(task+'_c')
        self.control_size = control_size
        self.template = getattr(templates, 'bdd100k_'+task+'_template') 
        if 'weather' in task:
            self.task = 'weather'
        elif 'scene' in task:
            self.task = 'scene'
        self.mode = mode
        if mode == 'train':
            self.populate_train()
            print('train dataset:', len(self.train_dataset))
        elif mode == 'val':
            self.populate_test(test_flag=False)
            print('val dataset:', len(self.test_dataset))
        elif mode == 'test':
            self.populate_test(test_flag=True)
            print('test dataset:', len(self.test_dataset))
    
    
    def populate_train(self):
        # /data/datasets/BDD100k/bdd100k/images/100k/ + val
        # /data/datasets/BDD100k/bdd100k/labels/det_20/
        if self.task == 'weather':
            train_csv = pd.read_csv('./data/csvs/bdd100k/bdd100k_train_train_labels_wea.csv', index_col=0)
        elif self.task == 'scene':
            train_csv = pd.read_csv('./data/csvs/bdd100k/bdd100k_train_train_labels_sce.csv', index_col=0)

        np.random.seed(2024)
        image_list, labels = [], []
        for i, cls_name in enumerate(self.control_classnames):
            cls_paths = train_csv[train_csv[self.task]==cls_name]['path'].values
            sampled_ids = np.random.choice(cls_paths.shape[0], 
                                          min(self.control_size, cls_paths.shape[0]),replace = False )
            image_list.extend( list(cls_paths[sampled_ids]) )
            labels.extend( [i]*len(sampled_ids) ) 
        
        root = os.path.join(self.location, 'images/100k/')
        self.train_dataset = BDD100k_Dataset(
            root, image_list, labels, transform=self.preprocess)
       
        if self.ct_sampler:
            sampler = ClassSampler(labels, self.ct_batch_size,  num_cls_per_batch = self.ct_num_cls_per_batch)
            shuffle = None
        else:
            sampler = None
            shuffle = True
            print('####: no ct-sampler')
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.ct_batch_size,
            shuffle=shuffle,
            sampler = sampler,
            num_workers=self.num_workers)

        ### test_loader for sequential iterate the whole dataset + training target for eval
        target_class = self.classnames[self.control_classnames.index('NA')]
        if self.task == 'weather':
            try:
                train_target = pd.read_csv(os.path.join('./data/csvs/bdd100k', f'bdd100k_train_{target_class}_labels_wea.csv'), index_col=0)
                target_paths = train_target['path'].to_list()  #train_csv[train_csv[self.task]=='foggy']['path'].to_list()
            except:
                target_paths = train_csv[train_csv[self.task]==target_class]['path'].to_list()
        elif self.task == 'scene':
            train_target = pd.read_csv(os.path.join('./data/csvs/bdd100k/', f'bdd100k_train_{target_class}_labels_sce.csv'), index_col=0)
            target_paths = train_target['path'].to_list()            
            
        image_list_plus_target = image_list + target_paths
        labels_plus_target = labels + [self.control_classnames.index('NA')]*len(target_paths)
        self.train_dataset_eval = BDD100k_Dataset(
            root, image_list_plus_target, labels_plus_target, transform=self.preprocess_eval )
        self.test_loader = torch.utils.data.DataLoader(
            self.train_dataset_eval,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )



    def populate_test(self, test_flag):
        if test_flag:
            if self.task=='scene':
                test_csv = pd.read_csv(os.path.join('./data/csvs/bdd100k/', f'bdd100k_val_labels_sce.csv'), index_col=0)
            else:
                test_csv = pd.read_csv(os.path.join('./data/csvs/bdd100k/', f'bdd100k_val_labels_wea.csv'), index_col=0)
        else:
            #### validation
            if self.task=='scene':
                test_csv = pd.read_csv(os.path.join('./data/csvs/bdd100k/', f'bdd100k_train_val_labels_sce.csv'), index_col=0)
            else:
                test_csv = pd.read_csv(os.path.join('./data/csvs/bdd100k/', f'bdd100k_train_val_labels_wea.csv'), index_col=0)
 
        ####remove unlabeled samples
        test_csv = test_csv.loc[ test_csv[self.task].isin(self.classnames) ]
        
        image_list = test_csv['path'].values
        for i, cls_name in enumerate(self.classnames):
            test_csv.loc[test_csv[self.task]==cls_name, self.task]=i
            
        labels = test_csv[self.task].values
        root = os.path.join(self.location, 'images/100k/')
        
        self.test_dataset = BDD100k_Dataset(
            root, image_list, labels, transform=self.preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)        


    def name(self):
        return 'bdd100k'
 
class BDD100k_Val(BDD100k):
    def __init__(self, preprocess,preprocess_eval=None, control_size=None, location='/data/datasets/BDD100k/bdd100k/', 
                 batch_size=128, ct_batch_size=50, ct_num_cls_per_batch = 5, ct_sampler = False, num_workers=12, task='weather'):
        super().__init__(preprocess,preprocess_eval, control_size, location, batch_size, ct_batch_size, ct_num_cls_per_batch, ct_sampler, num_workers, task, mode='val')  

class BDD100k_Test(BDD100k):
    def __init__(self, preprocess, preprocess_eval=None,  control_size=None, location='/data/datasets/BDD100k/bdd100k/', 
                 batch_size=128, ct_batch_size=50, ct_num_cls_per_batch=5, ct_sampler = False, num_workers=12, task='weather'):
        super().__init__(preprocess, preprocess_eval, control_size, location, batch_size, ct_batch_size, ct_num_cls_per_batch, ct_sampler, num_workers, task, mode='test')

    