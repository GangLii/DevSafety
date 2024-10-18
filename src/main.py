
import os
import numpy as np
import torch
import torch.nn as nn
from models.train import training
from src.models.modeling import ClassificationHead, CLIPEncoder
from src.args import parse_arguments
import logging
import random



def main(args):

    ###logging##################################################################
    os.makedirs(args.save + args.exp_name, exist_ok=True)
    args.save = args.save + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_seed" + str(args.seed)
    os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
    logging_path = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_seed" + str(args.seed)
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    assert args.save is not None, 'Please provide a path to store models'
    #############################################################################
    # Initialize the CLIP encoder
    clip_encoder = CLIPEncoder(args, keep_lang=True)
    
    ### add lora head
    if args.lora:
        transformer_width, embed_dim = clip_encoder.model.text_projection.size()
        clip_encoder.model.projection_fn_A = nn.Parameter(torch.empty(args.num_class, transformer_width, args.lora_dim))
        clip_encoder.model.projection_fn_B = nn.Parameter(torch.empty(args.num_class, args.lora_dim, embed_dim))
        
        nn.init.normal_(clip_encoder.model.projection_fn_A, std=transformer_width ** -0.5)
        nn.init.zeros_(clip_encoder.model.projection_fn_B)

    #######load trained model##########
    if args.base_model:
        state_dict = torch.load(args.base_model, map_location='cpu')
    
        ### without loading task heads
        filtered = {k:v for k,v in state_dict.items() if 'fn_' not in k}
        msg = clip_encoder.load_state_dict(filtered, strict=False)
        ### with loading task heads
        # msg = clip_encoder.load_state_dict(state_dict, strict=False)
        print ('missing_keys for finetuning: %s '%(msg.missing_keys))   
    if args.lora:  ### negative head        
        nn.init.normal_(clip_encoder.model.projection_fn_A[-1,:,:], std=transformer_width ** -0.5)
        nn.init.zeros_(clip_encoder.model.projection_fn_B[-1,:,:])
    #####################################

    logger.info(args)

    training(args, clip_encoder, logger)


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = parse_arguments()
    set_all_seeds(args.seed)
    main(args)
