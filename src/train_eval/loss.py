import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Optional
from torch import Tensor
import logging
import math
import numpy as np

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(image_features,
                    text_features,
                    local_loss=False,
                    gather_with_grad=False,
                    rank=0,
                    world_size=1,
                    use_horovod=False):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features



 
    
class SogCLR_Penalty(nn.Module):
    def __init__(self, N=10_000_000, num_ct_class=5, gamma2=0.8, temperature=0.07, 
                 beta = 40, tau = 0.1, total_epoch=40, cosine = False,
                 world_size=1, rank=0, h_negatives = 1,
                enable_surrogate=False, surrogate_c=1.0):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(SogCLR_Penalty, self).__init__()
        self.world_size = world_size
        self.rank = rank    
        self.total_epoch = total_epoch
        self.eps = torch.finfo(torch.float32).eps
        self.enable_surrogate = enable_surrogate
        self.c = surrogate_c # margin parameter for the square hinge loss  

        #### parameters for sogclr
        self.s_I = torch.zeros(N)#.cuda()
        self.s_T = torch.zeros(N)#.cuda()
        # self.b_I = torch.zeros(N).cuda()
        # self.b_T = torch.zeros(N).cuda()
        self.gamma1 = 0.8
        self.temperature = temperature
        self.h_negatives = h_negatives

        ##########parameters for constraint
        self.tau = tau
        self.beta = beta
        self.cosine = cosine
        self.u = torch.zeros(num_ct_class).cuda()
        self.gamma2 = gamma2


    def _sqh(self, x):
        return torch.max(torch.zeros_like(x), x + self.c) ** 2
    

    def forward(self, image_features, text_features, image_ids, text_ids, slabel, epoch,
                ### control set
                img_feas_c, txt_feas_c, labels_c, index_c, last_loss_c
                ):
        #Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]
        device = image_features.device
            
        if self.world_size > 1:
            image_features, text_features = gather_features(
                image_features, text_features, gather_with_grad=True)
 

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        pos_mask = (slabel == 1).squeeze()
        pos_image_ids = image_ids[pos_mask]
        pos_text_ids = text_ids[pos_mask]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = (sim - diag_sim[:, None])[pos_mask,:][:,~pos_mask] ##pos*neg
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = (sim - diag_sim[None, :])[~pos_mask,:][:, pos_mask] ##neg*pos
        
        if self.enable_surrogate:
            image_diffs = self._sqh(image_diffs)
            text_diffs = self._sqh(text_diffs)

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()
        
        # update b
        # old_b_I = self.b_I[image_ids]
        # new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        # self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        # old_b_T = self.b_T[text_ids]
        # new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        # self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]
        
        # exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * self.mask_neg # -b to avoid exp operation overflow
        # exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * self.mask_neg

        ##### 
        exp_image_diffs = torch.exp(image_diffs_d_temps) # pos*neg
        exp_text_diffs = torch.exp(text_diffs_d_temps) #*  neg *pos
 
        g_I = torch.mean(exp_image_diffs, dim=1, keepdim=True).detach()
        g_T = torch.mean(exp_text_diffs, dim=0, keepdim=True).detach()

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            # s_I = (1.0-self.gamma1) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma1 * g_I.squeeze()
            # s_T = (1.0-self.gamma1) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma1 * g_T.squeeze()
            ##### 
            s_I = (1.0-self.gamma1) * self.s_I[pos_image_ids].to(device)  + self.gamma1 * g_I.squeeze()
            s_T = (1.0-self.gamma1) * self.s_T[pos_text_ids].to(device)  + self.gamma1 * g_T.squeeze()

            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[pos_image_ids] = s_I.squeeze().cpu()
        self.s_T[pos_text_ids] = s_T.squeeze().cpu()
        
        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.mean(weights_image * image_diffs, dim=1, keepdim=True) 
        text_loss = torch.mean(weights_text * text_diffs, dim=0, keepdim=True) 

        contrastive_loss = image_loss.mean() + text_loss.mean()
        
        ###################################
        ##### constraint loss
        logits_per_image =  1/self.tau * img_feas_c @ txt_feas_c.T
        ce_loss = F.cross_entropy(logits_per_image,
                                    labels_c,
                                    reduction='none')
        
        ### aggregate by class
        unique_labels, labels_count = labels_c.unique(dim=0, return_counts=True)
        sum_by_cls = torch.zeros(self.u.shape[0], device=device).scatter_add_(
            0, labels_c,  ce_loss)
        mean_by_cls = sum_by_cls[unique_labels] / labels_count.float()  
        # loss from base model
        sum_by_cls_last = torch.zeros(self.u.shape[0], device=device).scatter_add_(
            0, labels_c,  last_loss_c)
        mean_by_cls_last = sum_by_cls_last[unique_labels] / labels_count.float() 
        constraint_value = mean_by_cls-mean_by_cls_last
        
        if self.u[unique_labels].sum()==0:
            self.u[unique_labels] = constraint_value.detach()
        else:
            self.u[unique_labels] = (1-self.gamma2)* self.u[unique_labels] + self.gamma2*constraint_value.detach()    
        
        ### cosine increasing
        if self.cosine:
            beta =  self.beta * 0.5 * (1. + math.cos(math.pi + math.pi * ( (epoch+1)/self.total_epoch ) ))  
        else:
            beta = self.beta
        constraint_loss = torch.maximum(beta * self.u[unique_labels],
                                        torch.zeros_like(mean_by_cls)).detach() *mean_by_cls*self.tau

        return contrastive_loss + constraint_loss.mean()
    


class SogCLR_Penalty_l1(nn.Module):
    def __init__(self, N=10_000_000, num_ct_class=5, gamma2=0.8, temperature=0.07, 
                 beta = 40, tau = 0.1, total_epoch=40, cosine = False,
                 world_size=1, rank=0, h_negatives = 1,
                enable_surrogate=False, surrogate_c=1.0):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(SogCLR_Penalty_l1, self).__init__()
        self.world_size = world_size
        self.rank = rank        
        ####sogclr
        self.s_I = torch.zeros(N)#.cuda()
        self.s_T = torch.zeros(N)#.cuda()
        # self.b_I = torch.zeros(N).cuda()
        # self.b_T = torch.zeros(N).cuda()
        self.gamma2 = gamma2
        self.gamma1 = 0.8
        self.temperature = temperature
        self.h_negatives = h_negatives
        ##########constraint
        self.tau = tau
        self.beta = beta
        self.cosine = cosine
        self.u = torch.zeros(num_ct_class).cuda()
        self.total_epoch = total_epoch
        
        self.eps = torch.finfo(torch.float32).eps
        self.enable_surrogate = enable_surrogate
        self.c = surrogate_c # margin parameter for the square hinge loss

    def _sqh(self, x):
        return torch.max(torch.zeros_like(x), x + self.c) ** 2
    

    def forward(self, image_features, text_features, image_ids, text_ids, slabel, epoch,
                ### control set
                img_feas_c, txt_feas_c, labels_c, index_c, last_loss_c
                ):
        #Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]
        device = image_features.device
            
        if self.world_size > 1:
            image_features, text_features = gather_features(
                image_features, text_features, gather_with_grad=True)
 

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        pos_mask = (slabel == 1).squeeze()
        pos_image_ids = image_ids[pos_mask]
        pos_text_ids = text_ids[pos_mask]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = (sim - diag_sim[:, None])[pos_mask,:][:,~pos_mask] ##pos*neg
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = (sim - diag_sim[None, :])[~pos_mask,:][:, pos_mask] ##neg*pos
        
        if self.enable_surrogate:
            image_diffs = self._sqh(image_diffs)
            text_diffs = self._sqh(text_diffs)

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()
        
        # update b
        # old_b_I = self.b_I[image_ids]
        # new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        # self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        # old_b_T = self.b_T[text_ids]
        # new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        # self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]
        
        # exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * self.mask_neg # -b to avoid exp operation overflow
        # exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * self.mask_neg

        ##### 
        exp_image_diffs = torch.exp(image_diffs_d_temps) # pos*neg
        exp_text_diffs = torch.exp(text_diffs_d_temps) #*  neg *pos
 
        g_I = torch.mean(exp_image_diffs, dim=1, keepdim=True).detach()
        g_T = torch.mean(exp_text_diffs, dim=0, keepdim=True).detach()


        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            # s_I = (1.0-self.gamma1) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma1 * g_I.squeeze()
            # s_T = (1.0-self.gamma1) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma1 * g_T.squeeze()
            ##### 
            s_I = (1.0-self.gamma1) * self.s_I[pos_image_ids].to(device)  + self.gamma1 * g_I.squeeze()
            s_T = (1.0-self.gamma1) * self.s_T[pos_text_ids].to(device)  + self.gamma1 * g_T.squeeze()

            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[pos_image_ids] = s_I.squeeze().cpu()
        self.s_T[pos_text_ids] = s_T.squeeze().cpu()
        
        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.mean(weights_image * image_diffs, dim=1, keepdim=True) 
        text_loss = torch.mean(weights_text * text_diffs, dim=0, keepdim=True) 

        contrastive_loss = image_loss.mean() + text_loss.mean()
        
        ###################################
        ##### constraint loss
        logits_per_image =  1/self.tau * img_feas_c @ txt_feas_c.T
        ce_loss = F.cross_entropy(logits_per_image,
                                    labels_c,
                                    reduction='none')
        
        ### aggregate by class
        unique_labels, labels_count = labels_c.unique(dim=0, return_counts=True)
        sum_by_cls = torch.zeros(self.u.shape[0], device=device).scatter_add_(
            0, labels_c,  ce_loss)
        mean_by_cls = sum_by_cls[unique_labels] / labels_count.float()  
        # loss from base model
        sum_by_cls_last = torch.zeros(self.u.shape[0], device=device).scatter_add_(
            0, labels_c,  last_loss_c)
        mean_by_cls_last = sum_by_cls_last[unique_labels] / labels_count.float() 
        constraint_value = mean_by_cls-mean_by_cls_last
        
        if self.u[unique_labels].sum()==0:
            self.u[unique_labels] = constraint_value.detach()
        else:
            self.u[unique_labels] = (1-self.gamma2)* self.u[unique_labels] + self.gamma2*constraint_value.detach()    
        
        ### cosine increasing
        if self.cosine:
            beta =  self.beta * 0.5 * (1. + math.cos(math.pi + math.pi * ( (epoch+1)/self.total_epoch ) ))  
        else:
            beta = self.beta
            
        # constraint_loss = torch.maximum(beta * self.u[unique_labels],
        #                                 torch.zeros_like(mean_by_cls)).detach() *mean_by_cls*self.tau
        constraint_loss = torch.where(self.u[unique_labels]>0,
                                        beta, 0).detach() *mean_by_cls*self.tau
                
            

        return contrastive_loss + constraint_loss.mean()
    
    
class SogCLR_RM(nn.Module):
    def __init__(self, N=10_000_000, num_ct_class=5,  gamma2=0.8, temperature=0.07, 
                 beta = 1, tau = 0.1, total_epoch=40,
                 world_size=1, rank=0,
                enable_surrogate=False, surrogate_c=1.0):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(SogCLR_RM, self).__init__()
        self.world_size = world_size
        self.rank = rank        
        ####sogclr
        self.s_I = torch.zeros(N)#.cuda()
        self.s_T = torch.zeros(N)#.cuda()
        # self.b_I = torch.zeros(N).cuda()
        # self.b_T = torch.zeros(N).cuda()
        self.gamma2 = gamma2
        self.gamma1 = 0.8
        self.temperature = temperature
        ##########constraint
        self.tau = tau
        self.beta = beta
        self.total_epoch = total_epoch
        self.num_ct_class= num_ct_class
        
        self.eps = torch.finfo(torch.float32).eps
        self.enable_surrogate = enable_surrogate
        self.c = surrogate_c # margin parameter for the square hinge loss

    def _sqh(self, x):
        return torch.max(torch.zeros_like(x), x + self.c) ** 2
    

    def forward(self, image_features, text_features, image_ids, text_ids, slabel, epoch,
                ### control set
                img_feas_c, txt_feas_c, labels_c, index_c,
                ):
        #Inputs:
        #    image_features, text_features is l2-normalized tensor
        #    image_features, text_features: [batch_size, emb_dim]
        device = image_features.device
            
        if self.world_size > 1:
            image_features, text_features = gather_features(
                image_features, text_features, gather_with_grad=True)
 

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)


        pos_mask = (slabel == 1).squeeze()
        pos_image_ids = image_ids[pos_mask]
        pos_text_ids = text_ids[pos_mask]


        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = (sim - diag_sim[:, None])[pos_mask,:][:,~pos_mask] ##pos*neg
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = (sim - diag_sim[None, :])[~pos_mask,:][:, pos_mask] ##neg*pos
        
        if self.enable_surrogate:
            image_diffs = self._sqh(image_diffs)
            text_diffs = self._sqh(text_diffs)

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()
        
        # update b
        # old_b_I = self.b_I[image_ids]
        # new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        # self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        # old_b_T = self.b_T[text_ids]
        # new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        # self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]
        
        # exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * self.mask_neg # -b to avoid exp operation overflow
        # exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * self.mask_neg

        ##### 
        exp_image_diffs = torch.exp(image_diffs_d_temps) # pos*neg
        exp_text_diffs = torch.exp(text_diffs_d_temps) #*  neg *pos
 
        g_I = torch.mean(exp_image_diffs, dim=1, keepdim=True).detach()
        g_T = torch.mean(exp_text_diffs, dim=0, keepdim=True).detach()


        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            # s_I = (1.0-self.gamma1) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamm1 * g_I.squeeze()
            # s_T = (1.0-self.gamma1) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma1 * g_T.squeeze()
            ##### 
            s_I = (1.0-self.gamma1) * self.s_I[pos_image_ids].to(device)  + self.gamma1 * g_I.squeeze()
            s_T = (1.0-self.gamma1) * self.s_T[pos_text_ids].to(device)  + self.gamma1 * g_T.squeeze()

            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[pos_image_ids] = s_I.squeeze().cpu()
        self.s_T[pos_text_ids] = s_T.squeeze().cpu()
        
        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.mean(weights_image * image_diffs, dim=1, keepdim=True) 
        text_loss = torch.mean(weights_text * text_diffs, dim=0, keepdim=True) 

        contrastive_loss = image_loss.mean() + text_loss.mean()
        
        ###################################
        ##### constraint loss
        logits_per_image =  1/self.tau * img_feas_c @ txt_feas_c.T
        ce_loss = F.cross_entropy(logits_per_image,
                                    labels_c,
                                    reduction='none')
        
        ### aggregate by class
        unique_labels, labels_count = labels_c.unique(dim=0, return_counts=True)
        sum_by_cls = torch.zeros(self.num_ct_class, device=device).scatter_add_(
            0, labels_c,  ce_loss)
        mean_by_cls = sum_by_cls[unique_labels] / labels_count.float()  

        ### cosine increasing
        # if self.cosine:
        #     beta =  self.beta * 0.5 * (1. + math.cos(math.pi + math.pi * ( (epoch+1)/self.total_epoch ) ))  
        # else:
        #     beta = self.beta
        
        # constraint_loss = torch.maximum(beta * self.u[unique_labels],
        #                                 torch.zeros_like(mean_by_cls)).detach() *mean_by_cls*self.tau
     

        return contrastive_loss + self.beta*mean_by_cls.mean()*self.tau    
    

    
    