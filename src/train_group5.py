import os
# from peft import LoraConfig, get_peft_model
import torch
import argparse
from datetime import datetime
from .model import SegVol
from .segment_anything_volumetric import sam_model_registry
import torch.multiprocessing as mp
import shutil
from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from .utils.loss import BCELoss, BinaryDiceLoss
from .data_utils import get_loader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import random
import numpy as np
from .model_group5 import AdaptedViT

import time

from peft import LoraConfig, get_peft_model


def setup_model(pretrained_model, lora_config):
    adapted_ViT = AdaptedViT()
    # adapted_ViT = get_peft_model(adapted_ViT, LoraConfig)
    # print(adapted_ViT)
    pretrained_state_dict = pretrained_model.model.image_encoder.state_dict()
    adapted_state_dict = adapted_ViT.state_dict()

    to_update = {module : weights for module, weights in pretrained_state_dict.items() if module in adapted_state_dict.keys() and adapted_state_dict[module].size() == weights.size()}

    print(f'Modules that are being updated : {to_update.keys()}')

    # Copy weights to our model
    adapted_state_dict.update(to_update)
    adapted_ViT.load_state_dict(adapted_state_dict)

    # Switch the image_encoders
    pretrained_model.model.image_encoder = adapted_ViT

    ViT = pretrained_model.model.image_encoder
    print('ViT before LoRA injection : ')
    print(ViT)
    total_params = sum(p.numel() for p in ViT.parameters())
    trainable = sum(p.numel() for p in ViT.parameters() if p.requires_grad == True)
    print(f'Number of parameters : {total_params}')
    print(f'Number of trainable parameters : {trainable}')
    print(f'Percentage trainable : {100*(trainable/total_params)}%')

    ViT = get_peft_model(ViT, lora_config)
    print('ViT after LoRA injection : ')
    print(ViT)
    total_params = sum(p.numel() for p in ViT.parameters())
    trainable = sum(p.numel() for p in ViT.parameters() if p.requires_grad == True)
    print(f'Number of parameters : {total_params}')
    print(f'Number of trainable parameters : {trainable}')
    print(f'Percentage trainable : {100*(trainable/total_params)}%')

    pretrained_model.model.image_encoder = ViT

    # Freeze everything
    for name, param in pretrained_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False 

    # Set new patch_embedding to trainable
    for param in pretrained_model.model.image_encoder.patch_embedding_equiv.parameters():
        param.requires_grad = True 
    
    # Set adapter module to trainable
    for param in pretrained_model.model.image_encoder.adapter.parameters():
        param.requires_grad = True 

    print('Final model : ')
    print(pretrained_model.model.image_encoder)
    total_params = sum(p.numel() for p in pretrained_model.model.parameters())
    trainable = sum(p.numel() for p in pretrained_model.model.parameters() if p.requires_grad == True)
    print(f'Number of parameters : {total_params}')
    print(f'Number of trainable parameters : {trainable}')
    print(f'Percentage trainable : {100*(trainable/total_params)}%')

    for name, param in pretrained_model.named_parameters():
        if param.requires_grad:
            print(name)

    return pretrained_model

def set_parse():
    parser = argparse.ArgumentParser()
    # %% set up parser
    parser.add_argument("--pretrain", type = str, default='')
    parser.add_argument("--resume", type = str, default='')
    parser.add_argument("--data_dir", type = str, default='')
    parser.add_argument("--dataset_codes", type = list, default=['0010', '0011'])
    # config
    parser.add_argument("--test_mode", default=False, type=bool)
    parser.add_argument("-infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("-spatial_size", default=(32, 256, 256), type=tuple)
    parser.add_argument("-patch_size", default=(4, 16, 16), type=tuple)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    parser.add_argument("--clip_ckpt", type = str, default = './config/clip')
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument('-num_workers', type=int, default=8)
    # dist
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--init_method', type = str, default = "env://")
    parser.add_argument('--bucket_cap_mb', type = int, default = 25,
                        help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
    # key params
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-weight_decay', type=float, default=1e-5)
    parser.add_argument('-warmup_epoch', type=int, default=10)
    parser.add_argument('-num_epochs', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument("--use_pseudo_label", default=True, type=bool)
    args = parser.parse_args()
    return args

def train_epoch(args, segvol_model, train_dataloader, optimizer, scheduler, epoch, iter_num):
    start = time.time()
    epoch_loss = 0
    epoch_sl_loss = 0
    epoch_ssl_loss = 0

    epoch_iterator = tqdm(
        train_dataloader, desc = "[RANK]", dynamic_ncols=True
    )
    
    for batch in epoch_iterator:
        image, gt3D = batch["image"].cuda(), batch["post_label"].cuda()
        pseudo_seg_cleaned = batch['pseudo_seg_cleaned'].cuda()
        organ_name_list = batch['organ_name_list']

        loss_step_avg = 0
        sl_loss_step_avg = 0
        ssl_loss_step_avg = 0
        for cls_idx in range(len(organ_name_list)):
            optimizer.zero_grad()
            organs_cls = organ_name_list[cls_idx]
            labels_cls = gt3D[:, cls_idx]

            if torch.sum(labels_cls) == 0:
                print(f'[RANK] ITER-{iter_num} --- No object, skip iter')
                continue
            
            segvol_model = segvol_model.to('cuda')
            
            print(image.shape)

            input_dict = {
                'image': image,
                'train_organs': organs_cls,
                'train_labels': labels_cls
            }

            sl_loss, ssl_loss = segvol_model(**input_dict)

            if args.use_pseudo_label:
                loss = sl_loss + 0.1 * ssl_loss
                ssl_loss_step_avg += ssl_loss.item()
                sl_loss_step_avg += sl_loss.item()
            loss_step_avg += loss.item()
            
            loss.backward()
            optimizer.step()
            print(f'[RANK] ITER-{iter_num} --- loss {loss.item()}, sl_loss, {sl_loss.item()}, ssl_loss {ssl_loss.item()}')
            iter_num += 1

        loss_step_avg /= len(organ_name_list)
        sl_loss_step_avg /= len(organ_name_list)
        ssl_loss_step_avg /= len(organ_name_list)
        print(f'[RANK] AVG loss {loss_step_avg}, sl_loss, {sl_loss_step_avg}, ssl_loss {ssl_loss_step_avg}')

        epoch_loss += loss_step_avg
        epoch_sl_loss += sl_loss_step_avg
        if args.use_pseudo_label:
            epoch_ssl_loss += ssl_loss_step_avg
    scheduler.step() 
    epoch_loss /= len(train_dataloader) + 1e-12
    epoch_ssl_loss /= len(train_dataloader) + 1e-12
    epoch_sl_loss /= len(train_dataloader) + 1e-12
    print(f'{args.model_save_path} ==> [RANK] ', 'epoch_loss: {}, ssl_loss: {}'.format(epoch_loss, epoch_ssl_loss))

    print(f'{epoch} took {time.time() - start} seconds.')
    return epoch_loss, iter_num

def main_worker(args):
    # Setup model
    
    clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
    segvol_model = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=False)
    segvol_model.model.text_encoder.tokenizer = clip_tokenizer

    # segvol_model = setup_model(segvol_model)

    # lora?
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["out_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    segvol_model = setup_model(segvol_model, lora_config)

    optimizer = torch.optim.AdamW(
        segvol_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.num_epochs)

    #%% train
    num_epochs = args.num_epochs
    iter_num = 0

    train_dataloader = get_loader(args)

    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_loss, iter_num = train_epoch(args, segvol_model, train_dataloader, optimizer, scheduler, epoch, iter_num)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        # save the model checkpoint
        if (epoch+1) % 10 == 0:
            checkpoint = {
                'model': segvol_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
                'epoch_loss': epoch_loss
            }
            torch.save(checkpoint, os.path.join(args.model_save_path, f'medsam_model_e{epoch+1}.pth'))

def main():
    # set seeds
    torch.manual_seed(2023)
    torch.cuda.empty_cache()
    args = set_parse()
    args.resume = "../Data/SegVol/weights/SegVol_v1.pth"
    args.num_workers = 1
    args.batch_size = 1
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.run_id = datetime.now().strftime("%Y%m%d-%H%M")
    
    # segvol model load
    model_save_path = os.path.join(args.work_dir, args.run_id)
    args.model_save_path = model_save_path
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12222'
    if args.use_pseudo_label:
        print('----- use pseudo_label -----')
    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)
    print("Spwaning processces, ngpus_per_node={}".format(ngpus_per_node))
    print(torch.cuda.device_count())
    print(args)
    print(f"=====> project save at {args.model_save_path}")
    # mp.spawn(main_worker, nprocs = ngpus_per_node, args=(ngpus_per_node, args))
    main_worker(args)


if __name__ == "__main__":
    main()

