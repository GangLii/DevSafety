from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm

import torch
import pandas as pd
import clip.clip as clip
from clip.loss import SogCLR_Penalty, SogCLR_RM,SogCLR_Penalty_l1

import src.templates as templates
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.utils import cosine_lr
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets
from torch.nn import functional as F

    
def training(args, clip_encoder, logger):
    assert args.train_data is not None, "Please provide a training dataset."
    logger.info(f'Fine-tuning Using {args.loss} Loss')
    model = clip_encoder
    clip_encoder.process_images = True
    print_every = 100

    img_text_data = get_data(
        args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess),
        epoch=0)
    assert len(img_text_data), 'At least one train or eval dataset must be specified.'
    ft_dataloader = img_text_data['train'].dataloader
    ft_iterator = iter(ft_dataloader)
    if args.batchs_per_epoch ==0:
        num_batches = ft_dataloader.num_batches
    else:
        num_batches = args.batchs_per_epoch
    print(f"Num batches is {num_batches}")
    
    model = model.cuda()

    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.loss in ['sog_class_semi', 'sog_class_semi_rm','sog_class_semi_l1']:
        ct_dataset_class = getattr(datasets, args.control_dataset)
        print("control dataset class", ct_dataset_class)
        print(f"Control dataset {args.control_dataset}")

        ct_dataset = ct_dataset_class(clip_encoder.train_preprocess,
                                preprocess_eval=clip_encoder.val_preprocess,
                                control_size = args.control_size,
                                location=args.ct_data_location,
                                batch_size = max(args.batch_size, args.ct_batch_size),
                                ct_batch_size=args.ct_batch_size,
                                ct_num_cls_per_batch = args.ct_num_cls_per_batch,
                                ct_sampler = args.ct_sampler,
                                num_workers=args.workers,
                                task=args.task,
                                mode='train', )
        ct_dataloader = ct_dataset.train_loader 
        ct_iterator = iter(ct_dataloader)
        
        #construct class text for bdd100k 
        template = ct_dataset.template #getattr(templates, args.template) 
        all_texts = []
        for classname in ct_dataset.classnames:
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts)  # tokenize
            all_texts.append(texts)

        all_texts = torch.stack(all_texts, dim=0)
        assert all_texts.shape[0] == len(ct_dataset.classnames)
        assert all_texts.shape[1] == len(template)
        assert all_texts.shape[2] == 77
        
        if args.loss in ['sog_class', 'sog_class_semi','sog_class_semi_l1']:
        ####compute previous loss values ahead to avoid dupicated computation
            pre_losses = torch.empty(len(ct_dataset.train_dataset_eval))
            pre_lables = torch.empty(len(ct_dataset.train_dataset_eval), dtype=torch.long)
            pre_preds = torch.empty(len(ct_dataset.train_dataset_eval), dtype=torch.long)
            model.eval()
            count = 0
            with torch.no_grad():
                for ct_image, ct_label, idx  in ct_dataset.test_loader: ###This test_loader iterate trainset sequentially
                    
                    ct_image, ct_label = ct_image.cuda(), ct_label.cuda()
                    if args.lora:
                        ct_image_features, ct_text_features, ct_logit_scale2 = model(
                            ct_image, all_texts[:, 0, :].cuda(), 
                            cls_ids=torch.arange(all_texts.shape[0], dtype=torch.long).cuda() )
                    else:
                        ct_image_features, ct_text_features, ct_logit_scale2 = model(
                            ct_image, all_texts[:, 0, :].cuda())
                    
                    logits = ct_image_features @ ct_text_features.T /args.tau   #### To do
                    pre_loss = F.cross_entropy(logits, ct_label, reduction='none')
                    pre_losses[idx] = pre_loss.detach().cpu()
                    pre_lables[idx] = ct_label.detach().cpu()
                    pre_preds[idx] = logits.argmax(dim=1).detach().cpu()
                    
                    count += len(ct_image)

            assert count == len(pre_losses), 'number doesnt match'




    if args.loss == 'sog_pnl':
        loss_fn = SogCLR_Penalty(num_ct_class=args.num_class, total_epoch=args.epochs,
                                    temperature=args.t,tau=args.tau, gamma2= args.gamma2,
                                    h_negatives= args.h_negatives,
                                    beta=args.beta, cosine=args.cosine)
    elif args.loss == 'sog_pnl_l1':
        loss_fn = SogCLR_Penalty_l1(num_ct_class=args.num_class, total_epoch=args.epochs,
                                    temperature=args.t,tau=args.tau, gamma2= args.gamma2,
                                    h_negatives= args.h_negatives,
                                    beta=args.beta, cosine=args.cosine)
    elif args.loss == 'sog_rm':
        loss_fn = SogCLR_RM(num_ct_class=args.num_class, total_epoch=args.epochs,
                                    temperature=args.t, tau=args.tau, gamma2= args.gamma2,
                                    beta=args.beta, )


    ###***
    if not args.lora:
        for key, param in model.named_parameters():
            if 'fn_' in key:
                torch.nn.init.zeros_(param)
                param.requires_grad = False
                
    ###***
    #model parameters --> optimizer
    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                          args.epochs * num_batches, args.min_lr)


    stats= []
    for epoch in range(-1, args.epochs):
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_flyp_loss_sum = 0
        model.train()
        model = model.cuda()
        if epoch != -1: # go to evaluation when epoch==-1
            # if epoch==0 and args.lora:  ### reset head    
            #     transformer_width, embed_dim = model.module.model.text_projection.size()
            #     torch.nn.init.normal_(model.module.model.projection_fn_A[5,:,:], std=transformer_width ** -0.5)
            #     torch.nn.init.zeros_(model.module.model.projection_fn_B[5,:,:])    #[-1,:,:]                
            for i in range(num_batches): 
                start_time = time.time()
                step = i + epoch * num_batches
                if epoch != -1:
                    scheduler(step)
                optimizer.zero_grad()

                try:
                    ft_batch = next(ft_iterator)
                except StopIteration:                    
                    ft_iterator = iter(ft_dataloader)
                    ft_batch = next(ft_iterator)


                ft_image, ft_text, image_ids, text_ids, label = ft_batch
                ft_image, ft_text = ft_image.cuda(), ft_text.cuda()

                
                if args.lora:
                    if args.pseudo_lable_key=='slabel':
                        ft_image_features, ft_text_features, logit_scale2 = model(
                            ft_image, ft_text, 
                            cls_ids= ( (ct_dataset.control_classnames.index('NA')+1)*label -1 ).cuda() )                
                else:
                    ft_image_features, ft_text_features, logit_scale2 = model(
                        ft_image, ft_text)
                    
                slabel = label==args.target_class
                

                if args.loss in ['sog_class_semi', 'sog_class_semi_rm','sog_class_semi_l1']:
                    try:
                        ct_batch = next(ct_iterator)
                    except StopIteration:
                        ct_iterator = iter(ct_dataloader)
                        ct_batch = next(ct_iterator)
                    ct_image, ct_label, c_ids = ct_batch
                    ct_image, ct_label = ct_image.cuda(), ct_label.cuda()
                        
                    if args.lora:
                        ct_image_features, ct_text_features, _ = model(
                            ct_image, all_texts[:, 0, :].cuda(),
                            cls_ids=torch.arange(all_texts.shape[0], dtype=torch.long).cuda())
                    else:
                        ct_image_features, ct_text_features, _ = model(
                            ct_image, all_texts[:, 0, :].cuda())
                    
                    #loading old ce loss values:
                    if args.loss in ['sog_class_semi','sog_class_semi_l1']:
                        pre_loss_c = pre_losses[c_ids].cuda()
                        ft_clip_loss = loss_fn(ft_image_features, ft_text_features,image_ids=image_ids, text_ids=text_ids, slabel=slabel, epoch=epoch, 
                                            img_feas_c=ct_image_features, txt_feas_c=ct_text_features, #text_featurec_c[6,emb_dim]
                                            labels_c = ct_label, #instancewise: [bsz,1]
                                            index_c=c_ids,
                                            last_loss_c = pre_loss_c,
                                            )
                    else: #['sog_class_semi_rm',]:                       
                        ft_clip_loss = loss_fn(ft_image_features, ft_text_features,image_ids=image_ids, text_ids=text_ids, slabel=slabel, epoch=epoch, 
                                            img_feas_c=ct_image_features, txt_feas_c=ct_text_features, #text_featurec_c[6,emb_dim]
                                            labels_c = ct_label, #instancewise: [bsz,1]
                                            index_c=c_ids,
                                            )


                ft_clip_loss.backward()
                optimizer.step()

                id_flyp_loss_sum += ft_clip_loss.item()

                if i % print_every == 0:
                    percent_complete = 100 * i / num_batches
                    logger.info(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                        f"ID FLYP Loss: {ft_clip_loss.item():.4f} ")


        id_flyp_loss_avg = id_flyp_loss_sum / num_batches
        print("loss:", id_flyp_loss_avg)


        classification_head = get_zeroshot_classifier(
            args, model.module.model)
        classification_head = classification_head.cuda()

        evaluate(model, args, classification_head,
                                epoch_stats, logger)
        
        
        logger.info(f"Avg ID FLYP Loss : {id_flyp_loss_avg:.4f}")
        epoch_stats['Avg ID FLYP Loss'] = round(id_flyp_loss_avg, 4)
        

        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
        
        # Saving model
        if False: 
            if args.save is not None:                
                os.makedirs(args.save, exist_ok=True)
                model_path = os.path.join(args.save, f'checkpoint_best_{epoch}.pt')
                logger.info('Saving model to' + str(model_path))
                model.module.save(model_path)
                # optim_path = os.path.join(args.save, f'optim_{epoch}.pt')
                # torch.save(optimizer.state_dict(), optim_path)        
        
        log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_seed" + str(args.seed)
        os.makedirs(log_dir, exist_ok=True)
        stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')


