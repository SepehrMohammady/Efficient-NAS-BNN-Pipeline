import argparse
import logging
import os
import os.path as osp
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

import models as models
# Ensure Cutout is correctly imported if used in transforms
from utils import CrossEntropyLossSmooth, KLLossSoft, get_logger, Cutout 

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Supernet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('logdir', metavar='DIR')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='superbnn_wakevision_large', # Updated for WakeVision dataset
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--dataset',
                    type=str,
                    default='WakeVision',
                    help='imagenet | cifar10 | WakeVision') # Added WakeVision
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs',
                    default=128,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warm_up', action='store_true')
parser.add_argument('--warm_up-multiplier', default=1, type=float)
parser.add_argument('--warm_up-epochs', default=5, type=int)
parser.add_argument('--cutout',
                    action='store_true',
                    default=False,
                    help='use cutout')
parser.add_argument('--cutout-length',
                    type=int,
                    default=16,
                    help='cutout length')
parser.add_argument('--label_smooth',
                    type=float,
                    default=0.1,
                    help='label smoothing')
parser.add_argument('-b',
                    '--batch-size',
                    default=512,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=2.5e-3,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
lr_scheduler_choice = ['StepLR', 'MultiStepLR', 'CosineAnnealingLR']
parser.add_argument('--lr-scheduler',
                    default='CosineAnnealingLR',
                    choices=lr_scheduler_choice)
parser.add_argument('--step-size',
                    default=30,
                    type=int,
                    help='step size of StepLR')
parser.add_argument('--gamma',
                    default=0.1,
                    type=float,
                    help='lr decay of StepLR or MultiStepLR')
parser.add_argument('--milestones',
                    default=[80, 120],
                    nargs='+',
                    type=int,
                    help='milestones of MultiStepLR')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=0, # Changed from 5e-6 in your log to 0 based on parser default
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('-d',
                    '--distill',
                    dest='distill',
                    action='store_true',
                    help='distill')
parser.add_argument('-p',
                    '--print-freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 100)')
parser.add_argument('--save-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='save frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size',
                    default=-1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')
# Add an argument for image size, used by WakeVision transforms
parser.add_argument('--img-size', default=224, type=int, help='input image size (square images assumed)')


def is_first_gpu(args, ngpus_per_node):
    return not args.multiprocessing_distributed or \
           (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)


def seed(seed_val=0): # Renamed parameter
    import os
    import random
    import sys
    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # torch.backends.cudnn.benchmark = False # Already in your code, good
    # torch.backends.cudnn.deterministic = True # Already in your code, good
    np.random.seed(seed_val)
    random.seed(seed_val)


def main():
    args = parser.parse_args()
    # seed(args.seed) # Moved to main_worker for DDP consistency

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    seed(args.seed) # Call seed at the beginning of each worker process
    args.gpu = gpu
    logger = None # Initialize
    writer = None # Initialize

    if args.gpu is not None:
        # Simplified print for initial GPU message
        if not args.distributed or (args.rank % ngpus_per_node == 0): # Only first process per node prints
            print(f'Use GPU: {args.gpu} for training')

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if is_first_gpu(args, ngpus_per_node):
        if not osp.exists(args.logdir):
            os.makedirs(args.logdir)
        logger = get_logger(name='Train',
                            log_file=osp.join(args.logdir, 'train.log'),
                            log_level=logging.INFO)
        logger.info(args)
        writer = SummaryWriter(osp.join(args.logdir, 'tf_logs'))
    
    if is_first_gpu(args, ngpus_per_node) and logger: # Check if logger is initialized
        logger.info(f"=> creating model '{args.arch}'")
    
    # Pass img_size to model constructor if the architecture function supports it
    # This requires model constructor functions (like superbnn_wakevision_large) to accept img_size
    if "wakevision" in args.arch.lower(): # Be more specific if needed
        model = models.__dict__[args.arch](img_size=args.img_size)
    else:
        model = models.__dict__[args.arch]()


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu) # For validation
    criterion_soft = CrossEntropyLossSmooth(args.label_smooth).cuda(args.gpu)
    criterion_kd = KLLossSoft().cuda(args.gpu)

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_transform = None
    val_transform = None

    if args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size), # Use args.img_size
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([
            transforms.Resize(int(args.img_size * 256 / 224)), # Standard ImageNet scaling
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(), normalize])
    elif args.dataset == 'cifar10':
        args.img_size = 32 # Override for CIFAR-10
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                         std=[0.24703233, 0.24348505, 0.26158768])
        train_transform = transforms.Compose([
            transforms.RandomCrop(args.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif args.dataset == 'WakeVision':
        # args.img_size should be set from command line (e.g. 128)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(), normalize])
    else:
        if logger: logger.error(f"Dataset {args.dataset} not implemented.")
        raise NotImplementedError(f"Dataset {args.dataset} not implemented in train.py")

    if args.cutout and train_transform is not None:
        train_transform.transforms.append(Cutout(args.cutout_length))

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, logger, writer, args) # Removed ngpus_per_node
        return

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if (p.ndimension() == 4 and 'conv' in pname) or \
           (p.ndimension() == 2 and ('linear' in pname or 'fc' in pname)):
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.Adam([
        {'params': other_parameters},
        {'params': weight_parameters, 'weight_decay': args.weight_decay}], lr=args.lr)
    
    scheduler_lr = get_lr_scheduler(optimizer, args)

    if args.resume:
        if os.path.isfile(args.resume):
            if is_first_gpu(args, ngpus_per_node) and logger:
                logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_lr.load_state_dict(checkpoint['scheduler'])
            if is_first_gpu(args, ngpus_per_node) and logger:
                logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            if is_first_gpu(args, ngpus_per_node) and logger:
                logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.warm_up: # Assuming warmup_epochs is correctly spelled in args
        args.milestones = [i - args.warm_up_epochs for i in args.milestones]


    optimizer.zero_grad()
    optimizer.step() # this zero gradient update is needed to avoid a warning message.

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Pass args to train function for DIAGNOSTIC prints
        train(train_loader, model, criterion_soft, criterion_kd, optimizer, epoch, logger, writer, args) # Removed ngpus_per_node

        scheduler_lr.step()
        
        # Validation can be done periodically if needed, e.g. every epoch or few epochs
        # For supernet training, validation is often skipped or done less frequently to save time
        # acc1_val, acc5_val = validate(val_loader, model, criterion, logger, writer, args)

        if is_first_gpu(args, ngpus_per_node): # Only first GPU saves checkpoints
            # if writer is not None and 'acc1_val' in locals(): # Check if validation was run
            #     writer.add_scalar('val/acc1', acc1_val, epoch)
            #     writer.add_scalar('val/acc5', acc5_val, epoch)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler_lr.state_dict(),
            }, args) # Pass full args
            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler_lr.state_dict(),
                }, args, name=f'epoch_{epoch + 1}.pth.tar') # Pass full args

    if is_first_gpu(args, ngpus_per_node) and writer is not None:
        writer.close()


def train(train_loader, model, criterion_soft, criterion_kd, optimizer, epoch, logger, writer, args): # Added args
    # ngpus_per_node is not directly used in train, is_first_gpu uses args.rank and ngpus_per_node from main_worker context
    # For simplicity if is_first_gpu is only for logging/writing, can pass it as a boolean flag
    first_gpu_flag = is_first_gpu(args, torch.cuda.device_count() if args.multiprocessing_distributed else 1)


    if first_gpu_flag and logger: # Check logger existence
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss(Rand)', ':.4e') # Renamed for clarity
        losses_l2 = AverageMeter('Loss_L2(Distill)', ':.4e')
        losses1 = AverageMeter('Loss1(Big)', ':.4e')
        losses2 = AverageMeter('Loss2(Small_KD)', ':.4e')
        top11 = AverageMeter('AccBig@1', ':6.2f')
        top15 = AverageMeter('AccBig@5', ':6.2f')
        top21 = AverageMeter('AccSmall@1', ':6.2f')
        top25 = AverageMeter('AccSmall@5', ':6.2f')
        top1 = AverageMeter('AccRand@1', ':6.2f')
        top5 = AverageMeter('AccRand@5', ':6.2f')
        progress_meters = [batch_time, data_time, losses1, losses2, losses, losses_l2, 
                           top11, top15, top21, top25, top1, top5]
        progress = ProgressMeter(logger, len(train_loader), progress_meters, prefix=f"Epoch: [{epoch}]")


    model.train()
    actual_model_module = model.module if hasattr(model, 'module') else model # Get actual model

    base_step = epoch * len(train_loader)
    if first_gpu_flag: end = time.time()

    for i, (images, target) in enumerate(train_loader):
        if first_gpu_flag: data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available(): # Should be args.gpu if set
            target = target.cuda(args.gpu if args.gpu is not None else 0, non_blocking=True)

        # Diagnostic prints from your previous version
        if i == 0 and epoch == args.start_epoch and first_gpu_flag and logger:
            logger.info(f"Diagnostic: Epoch {epoch}, Batch {i}")
            logger.info(f"Diagnostic: Model device: {next(model.parameters()).device if list(model.parameters()) else 'N/A'}")
            logger.info(f"Diagnostic: Images device: {images.device}")
            logger.info(f"Diagnostic: Target device: {target.device}")
            if hasattr(actual_model_module, 'biggest_cand'):
                 logger.info(f"Diagnostic: SuperBNN biggest_cand device: {actual_model_module.biggest_cand.device}")
            logger.info(f"Diagnostic: args.gpu = {args.gpu}")
            logger.info(f"Diagnostic: torch.cuda.current_device() = {torch.cuda.current_device()}")

        # Calculate effective batchsize for distributed training if used
        current_batch_size = images.size(0)
        if args.distributed: # Though for your case it's False
            batch_size_tensor = torch.tensor(current_batch_size, device=args.gpu if args.gpu is not None else 'cpu')
            dist.all_reduce(batch_size_tensor)
            total_batch_size_across_gpus = batch_size_tensor.item()
        else:
            total_batch_size_across_gpus = current_batch_size


        optimizer.zero_grad()

        actual_model_module.set_fp_weight()
        biggest_cand = actual_model_module.biggest_cand
        output_big, _ = model(images, biggest_cand) # model is potentially DDP wrapped
        loss1 = criterion_soft(output_big, target)
        loss1.backward()
        with torch.no_grad():
            soft_logits = output_big.clone().detach()

        acc11_val, acc15_val = accuracy(output_big, target, topk=(1, 5)) # Renamed vars

        actual_model_module.set_bin_weight()
        smallest_cand = actual_model_module.smallest_cand
        if args.distill: actual_model_module.open_distill()
        output_small, loss_l2 = model(images, smallest_cand)
        if args.distill:
            loss_l2.backward(retain_graph=True)
            actual_model_module.close_distill()
        
        loss2 = criterion_kd(output_small, soft_logits)
        loss2.backward()
        
        acc21_val, acc25_val = accuracy(output_small, target, topk=(1, 5)) # Renamed vars

        loss_rand_sum = torch.tensor(0.0, device=images.device) # Ensure on correct device
        loss_l2_total_rand = torch.tensor(0.0, device=images.device) # Ensure on correct device

        # For random paths, need to average gradients or handle updates carefully
        # Original code does loss.backward() for each random path
        for _ in range(2): # Original had idx, but not used
            cand = actual_model_module.get_random_cand()
            if args.distributed: dist.barrier(); dist.broadcast(cand, 0)
            if args.distill: actual_model_module.open_distill()
            output_rand, loss_l2_rand = model(images, cand)
            if args.distill:
                loss_l2_rand.backward(retain_graph=True)
                actual_model_module.close_distill()
            loss_r = criterion_kd(output_rand, soft_logits)
            loss_r.backward() # Accumulates gradients
            loss_rand_sum += loss_r
            loss_l2_total_rand += loss_l2_rand
        
        optimizer.step()

        acc1_rand, acc5_rand = accuracy(output_rand, target, topk=(1, 5)) # Accuracy of last random path

        if first_gpu_flag and logger:
            # Update AverageMeters (use .item() for losses, and ensure accs are single values)
            losses1.update(loss1.item(), current_batch_size)
            losses2.update(loss2.item(), current_batch_size)
            # Get float value from loss_l2, whether it's a tensor or already a float
            val_loss_l2 = loss_l2.item() if isinstance(loss_l2, torch.Tensor) else loss_l2
            
            # Get float value from loss_l2_total_rand
            val_loss_l2_total_rand = loss_l2_total_rand.item() if isinstance(loss_l2_total_rand, torch.Tensor) else loss_l2_total_rand
            
            losses_l2.update(val_loss_l2 + val_loss_l2_total_rand, current_batch_size)
            losses.update(loss_rand_sum.item(), current_batch_size) # Sum of random path losses
            
            top1.update(acc1_rand[0].item() if isinstance(acc1_rand, list) else acc1_rand.item(), total_batch_size_across_gpus)
            top5.update(acc5_rand[0].item() if isinstance(acc5_rand, list) and len(acc5_rand)>1 else (acc5_rand[1].item() if isinstance(acc5_rand, list) and len(acc5_rand)>1 else 0.0) , total_batch_size_across_gpus)
            top11.update(acc11_val[0].item(), total_batch_size_across_gpus)
            top15.update(acc15_val[1].item() if len(acc15_val)>1 else 0.0, total_batch_size_across_gpus)
            top21.update(acc21_val[0].item(), total_batch_size_across_gpus)
            top25.update(acc25_val[1].item() if len(acc25_val)>1 else 0.0, total_batch_size_across_gpus)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0: # Simpler condition, original was (i%p==0 and i>0)
                progress.display(i + 1) # Display with current batch number (1-indexed)
                if writer is not None:
                    current_iter = base_step + i
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], current_iter)
                    writer.add_scalar('train/AccRand@1', top1.avg, current_iter)
                    writer.add_scalar('train/AccBig@1', top11.avg, current_iter)
                    writer.add_scalar('train/AccSmall@1', top21.avg, current_iter)
                    writer.add_scalar('train/LossRand', losses.avg, current_iter)
                    writer.add_scalar('train/LossBig', losses1.avg, current_iter)
                    writer.add_scalar('train/LossSmallKD', losses2.avg, current_iter)
                    writer.add_scalar('train/LossL2Distill', losses_l2.avg, current_iter)


def validate(val_loader, model, criterion, logger, writer, args): # Added args
    first_gpu_flag = is_first_gpu(args, torch.cuda.device_count() if args.multiprocessing_distributed else 1)
    if first_gpu_flag and logger:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(logger, len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    model.eval()
    with torch.no_grad():
        if first_gpu_flag: end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu if args.gpu is not None else 0, non_blocking=True)

            output, _ = model(images, actual_model_module.biggest_cand if hasattr(actual_model_module, 'biggest_cand') else None) # Eval with biggest arch
            loss = criterion(output, target)
            
            acc1_val, acc5_val = accuracy(output, target, topk=(1, 5)) # Renamed

            # For distributed, need to aggregate batchsize and accs correctly
            current_batch_size = images.size(0)
            if args.distributed:
                # Example of how one might gather results in DDP (simplified)
                # This part needs careful implementation for DDP if used.
                # For single GPU, total_batch_size_across_gpus = current_batch_size
                pass # Placeholder for DDP gathering

            total_batch_size_across_gpus = current_batch_size # Assuming single GPU for now

            if first_gpu_flag and logger:
                losses.update(loss.item(), current_batch_size)
                top1.update(acc1_val[0].item(), total_batch_size_across_gpus)
                top5.update(acc5_val[1].item() if len(acc5_val)>1 else 0.0, total_batch_size_across_gpus)
                batch_time.update(time.time() - end)
                end = time.time()
                if i % args.print_freq == 0:
                    progress.display(i + 1)
        
        if first_gpu_flag and logger:
            logger.info(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
            if writer is not None: # Log to TensorBoard if writer is available
                writer.add_scalar('val/Acc@1', top1.avg, args.start_epoch) # Use current epoch from args
                writer.add_scalar('val/Acc@5', top5.avg, args.start_epoch)


    if first_gpu_flag: return top1.avg, top5.avg
    else: return torch.tensor(-1.0), torch.tensor(-1.0) # Return tensors for consistency


def save_checkpoint(state, args, name='checkpoint.pth.tar'): # Added args
    # filename determined by args.logdir passed from main_worker
    filename = osp.join(args.logdir, name)
    torch.save(state, filename)


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        if self.count > 0: self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, logger, num_batches, meters, prefix=""):
        self.logger = logger
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger: self.logger.info('\t'.join(entries)) # Check if logger exists
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_lr_scheduler(optimizer, args):
    scheduler_after_warmup = None
    if args.lr_scheduler == 'CosineAnnealingLR':
        # print('Use cosine scheduler') # Printed in orchestrator or Cell 2
        scheduler_after_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - (args.warm_up_epochs if args.warm_up else 0) # Adjust T_max if warmup
        )
    elif args.lr_scheduler == 'StepLR':
        scheduler_after_warmup = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        scheduler_after_warmup = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)
    else:
        raise NotImplementedError
    
    if args.warm_up:
        # print('Use warm_up scheduler') # Printed in orchestrator or Cell 2
        return GradualWarmupScheduler( 
            optimizer,
            multiplier=args.warm_up_multiplier,
            total_epoch=args.warm_up_epochs, # This is the number of WARMUP epochs
            after_scheduler=scheduler_after_warmup)
    else:
        return scheduler_after_warmup


# VVVVVV MODIFIED ACCURACY FUNCTION VVVVVV
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        num_classes = output.size(1)
        
        # Determine valid k values based on num_classes
        valid_topk_values = []
        for k_val_requested in topk:
            if k_val_requested <= num_classes:
                valid_topk_values.append(k_val_requested)
        
        if not valid_topk_values: # If no k is valid (e.g. topk=(5,) for num_classes=2)
            if 1 <= num_classes: # Default to top-1 if possible
                valid_topk_values = [1]
            else: # Should not happen if num_classes >= 1
                  # Return list of tensors with 0.0 of same length as original topk
                return [torch.tensor(0.0, device=output.device) for _ in topk]

        maxk_to_compute = max(valid_topk_values)
        batch_size_acc = target.size(0) # Renamed batch_size

        _, pred = output.topk(maxk_to_compute, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results_list = []
        for k_original_requested in topk: # Iterate through the originally requested k values
            if k_original_requested in valid_topk_values:
                # This k is valid and was computed up to maxk_to_compute
                # We need to slice `correct` appropriately for this specific k_original_requested
                correct_k = correct[:k_original_requested].reshape(-1).float().sum(0, keepdim=True)
                results_list.append(correct_k.mul_(100.0 / batch_size_acc))
            else:
                # This k was invalid (e.g., top-5 for 2 classes), append a dummy tensor (e.g., 0.0)
                results_list.append(torch.tensor([0.0], device=output.device)) # Ensure it's a tensor with one element
        return results_list
# ^^^^^^ END OF MODIFIED ACCURACY FUNCTION ^^^^^^


if __name__ == '__main__':
    main()