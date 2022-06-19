import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler

# CUDA_VISIBLE_DEVICES=4,5,6,7 python linear_eval_hardneg.py --data_dir /home/minkyu/data/imagenet100/ --log_dir ./logs/ -c configs/simsiam_image100_eval_sgd_hardneg.yaml --ckpt_dir ./cache --hide_progress --eval_from ./cache/simsiam-imagenet100-experiment-resnet50_final.pth

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    
    lr = args.eval.base_lr #*args.eval.batch_size/256 #args.lr
    for milestone in [30,40,50]:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    train_dataset = get_dataset( 
            transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs), 
            train=True, 
            **args.dataset_kwargs
        )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=(args.eval.batch_size//args.gpus),
        shuffle=False,
        sampler = train_sampler,
        **args.dataloader_kwargs
    )

    test_dataset = get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=(args.eval.batch_size//args.gpus),
        shuffle=False,
        **args.dataloader_kwargs
    )


    model = get_backbone(args.model.backbone)
    classifier = nn.Linear(in_features=model.output_dim, out_features=100, bias=True).to(args.device)

    assert args.eval_from is not None
    save_dict = torch.load(args.eval_from, map_location='cpu')
    msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
    
    # print(msg)
    model = model.to(args.device)
    # model = torch.nn.DataParallel(model)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    # if torch.cuda.device_count() > 1: 
     
    classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = DDP(classifier, device_ids=[gpu], find_unused_parameters=True)
    # define optimizer
    optimizer = get_optimizer(
        args.eval.optimizer.name, classifier, 
        lr=args.eval.base_lr*args.eval.batch_size/256, 
        momentum=args.eval.optimizer.momentum, 
        weight_decay=args.eval.optimizer.weight_decay)

    # define lr scheduler
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr*args.eval.batch_size/256, 
        args.eval.num_epochs, args.eval.base_lr*args.eval.batch_size/256, args.eval.final_lr*args.eval.batch_size/256, 
        len(train_loader),
    )

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)
        
        for idx, (images, labels) in enumerate(local_progress):

            classifier.zero_grad()
            with torch.no_grad():
                feature = model(images.to(args.device))
            lr = adjust_learning_rate(optimizer, epoch, args)
            preds = classifier(feature)

            loss = F.cross_entropy(preds, labels.to(args.device))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            # lr = lr_scheduler.step()
            local_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg})

        if gpu==0 and (epoch+1) == (args.eval.num_epochs-1):
            print('epoch:',epoch+1)
            classifier.eval()
            correct, total = 0, 0
            acc_meter.reset()
            if gpu == 0:
                for idx, (images, labels) in enumerate(test_loader):
                    with torch.no_grad():
                        feature = model(images.to(args.device))
                        preds = classifier(feature).argmax(dim=1)
                        correct = (preds == labels.to(args.device)).sum().item()
                        acc_meter.update(correct/preds.shape[0])
                print(f'Accuracy = {acc_meter.avg*100:.2f}')
            break



if __name__ == "__main__":
    args = get_args()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "3569"
    args.world_size = args.gpus * args.nodes
    mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    # main(args=get_args())

















