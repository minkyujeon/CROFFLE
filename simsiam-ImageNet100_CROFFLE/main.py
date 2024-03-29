import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --data_dir /home/minkyu/data/imagenet/ --log_dir ./logs/ -c configs/simsiam_imagenet.yaml --ckpt_dir ./imagenet_simsiam_jin_corrected/

def cleanup():
    dist.destroy_process_group()

def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    train_dataset = get_dataset(transform=get_aug(train=True, **args.aug_kwargs), train=True, **args.dataset_kwargs)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=False,
        batch_size=(args.train.batch_size // args.gpus),
        sampler = train_sampler,
        **args.dataloader_kwargs
    )

    memory_dataset = get_dataset(transform=get_aug(train=False,train_classifier=False, **args.aug_kwargs), memory='True', train=True, **args.dataset_kwargs)


    memory_loader = torch.utils.data.DataLoader(
        dataset=memory_dataset,
        shuffle=False,
        batch_size=(args.train.batch_size // args.gpus),
        **args.dataloader_kwargs
    )

    test_datset = get_dataset( transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), train=False,**args.dataset_kwargs)
    # print('train_dataset:',len(train_dataset))
    # print('memory_dataset:',len(memory_dataset))
    # print('test_dataset:',len(test_datset))

    test_loader = torch.utils.data.DataLoader(
        dataset= test_datset,
        shuffle=False,
        batch_size=(args.train.batch_size // args.gpus),
        **args.dataloader_kwargs
    )
    print("BAtch size:",(args.train.batch_size // args.gpus))
    # define model
    model = get_model(args.model).cuda(gpu)
    # print('model:',model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )
    if gpu ==0:
        logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0 
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        model.train()

        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels) in enumerate(local_progress):

            model.zero_grad()
            data_dict, deta_dict2 = model.forward(images1.cuda(non_blocking=True), images2.cuda(non_blocking=True))
            loss = data_dict['loss'] # ddp
            L_inter = data_dict['L_rand']
            # loss = 0.9*loss + 0.1*L_inter
            
            (loss+L_inter).backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            local_progress.set_postfix(data_dict)
            if gpu ==0:
                logger.update_scalers(data_dict)
            if gpu==0 and idx%200 == 0:
                print('pos1:',deta_dict2['pos1'], 'neg1:',deta_dict2['neg1'], 'neg2:',deta_dict2['neg2'])
                print('pos2:',deta_dict2['pos2'], 'neg3:',deta_dict2['neg3'], 'neg4:',deta_dict2['neg4'])
                print('-'*30)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0 and gpu==0: 
            accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, gpu, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress) 
        
        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        global_progress.set_postfix(epoch_dict)

        if gpu == 0:
            logger.update_scalers(epoch_dict)
   
        # Save checkpoint
        if gpu ==0 and epoch % args.train.knn_interval == 0:
            model_path = os.path.join(args.ckpt_dir, f"{args.name}_{epoch+1}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.module.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")
            with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
                f.write(f'{model_path}')
        
    if gpu == 0: # and (epoch+1) == (args.train.stop_at_epoch-1):
        model_path = os.path.join(args.ckpt_dir, f"{args.name}_{epoch+1}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.module.state_dict()
        }, model_path)
        print(f"Final Model saved to {model_path}")
        with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
            f.write(f'{model_path}')


    if gpu ==0:
        logger.close()

    if args.eval is not False and gpu == 0:
        args.eval_from = model_path
        linear_eval(args)


if __name__ == "__main__":
    args = get_args()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8441"
    args.world_size = args.gpus * args.nodes

    # Initialize the process and join up with the other processes.
    # This is “blocking,” meaning that no process will continue until all processes have joined.

    mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')



    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














