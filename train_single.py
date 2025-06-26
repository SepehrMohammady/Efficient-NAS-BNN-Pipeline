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
from utils import Cutout, get_logger, tuple2cand

# import wandb

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('checkpoint',
                    type=str,
                    metavar='PATH',
                    help='path to searched checkpoint')
parser.add_argument('logdir', metavar='DIR')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='superbnn_wakevision_large',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: superbnn_cifar10)')
parser.add_argument('--dataset',
                    type=str,
                    default='WakeVision',
                    help='imagenet | cifar10 | WakeVision')
parser.add_argument('--img-size', type=int, default=None, # <<< ADDED IMG_SIZE ARGUMENT
                    help='Input image size (square), for dummy_input in to_static. Required for non-fixed size datasets like WakeVision.')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--ops', type=int, default=80)
parser.add_argument('--epochs',
                    default=25,
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
                    default=1e-5,
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
                    default=0,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 100)')
parser.add_argument('--save-freq',
                    default=1,
                    type=int,
                    metavar='N',
                    help='save frequency (default: 1)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
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


def is_first_gpu(args, ngpus_per_node):
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parser.parse_args()
    seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    seed(args.seed) # Call seed for reproducibility in each worker
    args.gpu = gpu
    logger = None # Initialize
    writer = None # Initialize

    if args.gpu is not None:
        # Simplified print, logger might not be ready for non-master DDP processes
        if not args.distributed or (args.distributed and (args.rank == 0 or (args.rank * ngpus_per_node + gpu == 0))):
            print(f'Use GPU: {args.gpu} for training')

    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if is_first_gpu(args, ngpus_per_node):
        if not osp.exists(args.logdir):
            os.makedirs(args.logdir, exist_ok=True)
        logger = get_logger(name='Train', log_file=osp.join(args.logdir, 'train.log'),
                            log_level=logging.INFO)
        logger.info(args)
        # writer = SummaryWriter(osp.join(args.logdir, 'tf_logs')) # Uncomment if using TensorBoard
        # wandb.init(...) # Uncomment if using wandb

    # Load searched architecture
    if logger and is_first_gpu(args, ngpus_per_node): logger.info(f"=> Loading search info from: {args.checkpoint}")
    searched_info = torch.load(args.checkpoint, map_location='cpu') # args.checkpoint is info.pth.tar
    
    if args.ops not in searched_info['pareto_global']:
        errmsg = f"OPs key {args.ops} not found in Pareto front of {args.checkpoint}. Available keys: {list(searched_info['pareto_global'].keys())}"
        if logger: logger.error(errmsg)
        else: print(f"ERROR: {errmsg}")
        return # Exit if key not found
        
    arch_tuple = searched_info['pareto_global'][args.ops]
    arch_cand_tensor = tuple2cand(arch_tuple)

    # Create model with specific architecture
    if logger and is_first_gpu(args, ngpus_per_node): logger.info(f"=> Creating model '{args.arch}' with specific sub_path")
    model = models.__dict__[args.arch](sub_path=arch_cand_tensor)

    # Load pretrained supernet weights
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            if logger and is_first_gpu(args, ngpus_per_node):
                logger.info("=> Loading supernet weights from '{}'".format(args.pretrained))
            supernet_checkpoint = torch.load(args.pretrained, map_location='cpu')
            
            supernet_state_dict_cleaned = {}
            for k, v in supernet_checkpoint['state_dict'].items():
                supernet_state_dict_cleaned[k.replace('module.', '')] = v
            
            # Load supernet weights into the specific architecture instance.
            # strict=False is important as the sub-architecture only uses a subset of supernet weights.
            model.load_state_dict(supernet_state_dict_cleaned, strict=False) 
            if logger and is_first_gpu(args, ngpus_per_node):
                logger.info("=> Loaded supernet weights from '{}'".format(args.pretrained))
        else:
            if logger and is_first_gpu(args, ngpus_per_node):
                logger.info("=> NO supernet checkpoint found at '{}'".format(args.pretrained))
            # Depending on desired behavior, you might exit or proceed with random init for the specific arch
            # For fine-tuning from supernet, this is usually a critical error.
            # exit(1) # Or handle as appropriate
    else:
        if logger and is_first_gpu(args, ngpus_per_node):
            logger.info("=> No --pretrained supernet specified. Model will use its default initialization for the sub_path.")


    # --- Dummy input creation for to_static ---
    dummy_input_size = 0
    if args.dataset == 'imagenet':
        dummy_input_size = getattr(args, 'img_size', 224)
    elif args.dataset == 'cifar10':
        dummy_input_size = 32
    elif args.dataset == 'WakeVision':
        if args.img_size is None:
            args.img_size = 128 # Default if not passed
            warn_msg = f"--img-size not provided for WakeVision in train_single.py, defaulting to {args.img_size}. Ensure this matches model config."
            if logger and is_first_gpu(args, ngpus_per_node): logger.warning(warn_msg)
            elif is_first_gpu(args, ngpus_per_node): print(f"Warning: {warn_msg}")
        dummy_input_size = args.img_size
    else:
        err_msg_dataset = f"Unknown dataset '{args.dataset}' for dummy_input creation in train_single.py"
        if logger and is_first_gpu(args, ngpus_per_node): logger.error(err_msg_dataset)
        raise NotImplementedError(err_msg_dataset)
    
    dummy_input = torch.randn((1, 3, dummy_input_size, dummy_input_size))

    # --- Model to static and GPU placement ---
    # Move model to GPU BEFORE calling to_static, so to_static creates static params on GPU
    target_device_for_model = torch.device('cpu')
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        target_device_for_model = torch.device(f'cuda:{args.gpu}')
        if logger and is_first_gpu(args, ngpus_per_node): logger.info(f"Model moved to {target_device_for_model} before to_static.")
    # Not handling DDP/DataParallel here as train_single is usually for one arch on one GPU
    # If DDP/DP were used for fine-tuning, model would be wrapped after to_static.
    
    dummy_input = dummy_input.to(target_device_for_model) # Ensure dummy_input is on the same device as model
    
    model.eval() # Set to eval before to_static
    # SuperBNN.to_static uses the sub_path it was initialized with (arch_cand_tensor)
    model.to_static(dummy_input) 
    if logger and is_first_gpu(args, ngpus_per_node): logger.info("Model converted to static form.")

    # Log FLOPs/OPs for the static model
    # get_ops should work correctly now as model has self.sub_path_config_for_static_model set
    flops, bitops, total_ops = model.get_ops() 
    if logger and is_first_gpu(args, ngpus_per_node):
        # logger.info(f"=> creating model '{args.arch}'") # Already created
        logger.info(f"=> Static arch to fine-tune (from key {args.ops}): '{arch_tuple}'") # Use arch_tuple for logging
        logger.info('=> FLOPs {:.4f}M, BitOPs {:.4f}M, Total OPs (FLOPs + BitOPs/64) {:.4f}M'.format(
            flops, bitops, total_ops))

    # Re-apply GPU settings after to_static if model was recreated or device lost (unlikely but safe)
    if not torch.cuda.is_available():
        if logger and is_first_gpu(args, ngpus_per_node): logger.info('Using CPU, this will be slow for fine-tuning')
        elif is_first_gpu(args, ngpus_per_node): print('Using CPU, this will be slow for fine-tuning')
    elif args.distributed: # This block is for DDP fine-tuning - likely not your case for train_single
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            # model.cuda(args.gpu) # Already on GPU
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            # model.cuda() # Already on GPU
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        # model.cuda(args.gpu) # Already on GPU
    # else: DataParallel fallback, not typical for train_single after specific GPU set

    criterion = nn.CrossEntropyLoss().cuda(args.gpu if args.gpu is not None and torch.cuda.is_available() else 'cpu')

    # --- Data loading code for Fine-tuning ---
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    
    train_transform = None
    val_transform = None
    current_loader_img_size = getattr(args, 'img_size', 128 if args.dataset == 'WakeVision' else (224 if args.dataset == 'imagenet' else 32) )


    if args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        train_transform = transforms.Compose([transforms.RandomResizedCrop(current_loader_img_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(int(current_loader_img_size*256/224)), transforms.CenterCrop(current_loader_img_size), transforms.ToTensor(), normalize])
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif args.dataset == 'WakeVision':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        train_transform = transforms.Compose([
            transforms.Resize((current_loader_img_size, current_loader_img_size)), 
            transforms.RandomHorizontalFlip(), 
            # Add more augmentations suitable for fine-tuning WakeVision here if needed
            transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([
            transforms.Resize((current_loader_img_size, current_loader_img_size)), 
            transforms.ToTensor(), normalize])
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented in train_single.py dataloader.")

    if args.cutout and train_transform is not None: # Add cutout if specified and transform exists
        train_transform.transforms.append(Cutout(args.cutout_length))

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)

    # ... (Sampler and DataLoader setup as in your original, using args.batch_size and args.workers directly here as DDP division is handled above IF DDP enabled) ...
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=args.rank, num_replicas=args.world_size)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, rank=args.rank, num_replicas=args.world_size, shuffle=False) # No shuffle for val
    else:
        train_sampler = None
        val_sampler = None # No sampler for val if not distributed

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, # No shuffle for val_loader
        num_workers=args.workers, pin_memory=True, sampler=val_sampler) # Use val_sampler

    # Optimizer setup (as per your original train_single.py)
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
        {'params': weight_parameters, 'weight_decay': args.weight_decay}],
        lr=args.lr)
    scheduler_lr = get_lr_scheduler(optimizer, args) # Assuming get_lr_scheduler is defined in train_single.py

    # Resume logic for fine-tuning checkpoint itself
    if args.resume:
        if os.path.isfile(args.resume):
            if logger and is_first_gpu(args, ngpus_per_node):
                logger.info(f"=> loading fine-tuning checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}' if args.gpu is not None else 'cpu'
            checkpoint_resume = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint_resume['epoch']
            # If model is DDP wrapped, checkpoint keys might have 'module.'
            # The model here might or might not be DDP wrapped at this point.
            # Safest to handle 'module.' when loading.
            state_dict_resume = checkpoint_resume['state_dict']
            if not args.distributed and any(k.startswith('module.') for k in state_dict_resume.keys()):
                 new_state_dict_resume = {k[7:]: v for k,v in state_dict_resume.items()}
                 model.load_state_dict(new_state_dict_resume)
            elif args.distributed and not any(k.startswith('module.') for k in state_dict_resume.keys()) and hasattr(model, 'module'):
                 # model is DDP, checkpoint is not. Add 'module.'
                 new_state_dict_resume = {'module.'+k: v for k,v in state_dict_resume.items()}
                 model.load_state_dict(new_state_dict_resume)
            else: # DDP and checkpoint DDP, or no DDP and no DDP in checkpoint
                 model.load_state_dict(state_dict_resume)

            optimizer.load_state_dict(checkpoint_resume['optimizer'])
            scheduler_lr.load_state_dict(checkpoint_resume['scheduler'])
            if logger and is_first_gpu(args, ngpus_per_node):
                logger.info(f"=> loaded fine-tuning checkpoint '{args.resume}' (epoch {args.start_epoch})")
        else:
            if logger and is_first_gpu(args, ngpus_per_node):
                logger.info(f"=> no fine-tuning checkpoint found at '{args.resume}', starting from supernet weights.")
    
    if args.evaluate: # Evaluate before training if flag is set
        validate(val_loader, model, criterion, logger, writer, ngpus_per_node, args) # Assuming validate takes writer
        return

    # Warmup scheduler adjustment
    if args.warm_up and args.start_epoch < args.warm_up_epochs: # Only adjust if still in warmup phase
        # This might need more careful handling if resuming from within warmup
        if logger and is_first_gpu(args, ngpus_per_node): logger.info("Adjusting warmup scheduler based on start_epoch.")
        # The GradualWarmupScheduler might need specific handling for resuming.
        # For simplicity, if resuming into warmup, it might just continue.
        # If resuming after warmup, milestones should have been adjusted already.
        pass # No explicit adjustment here, assume scheduler_lr handles it with load_state_dict
    elif args.warm_up and args.start_epoch == 0 : # Only do this if truly starting from epoch 0 of fine-tuning
         args.milestones = [m_stone - args.warm_up_epochs for m_stone in args.milestones if m_stone > args.warm_up_epochs]


    optimizer.zero_grad() # From original train_single
    optimizer.step()      # From original train_single

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Call your 'train' function for one epoch of fine-tuning
        train(train_loader, model, criterion, optimizer, epoch, logger, writer, ngpus_per_node, args) # Assuming train takes writer
        scheduler_lr.step()

        acc1_val, acc5_val = validate(val_loader, model, criterion, logger, writer, ngpus_per_node, args) # Assuming validate takes writer

        if is_first_gpu(args, ngpus_per_node):
            if writer is not None:
                writer.add_scalar('val/acc1', acc1_val, epoch)
                writer.add_scalar('val/acc5', acc5_val, epoch)
            
            current_checkpoint_data = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'sub_path_tuple': arch_tuple, # Save the architecture tuple for reference
                'ops_key': args.ops,         # Save the OPs key
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler_lr.state_dict(),
                'val_acc1': acc1_val, # Save last validation accuracy
            }
            save_checkpoint(current_checkpoint_data, args) # Overwrites checkpoint.pth.tar
            if (epoch + 1) % args.save_freq == 0: # Save_freq is 1 in your case
                save_checkpoint(current_checkpoint_data, args, name=f'epoch_{epoch + 1}.pth.tar')
    
    if is_first_gpu(args, ngpus_per_node) and writer is not None:
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch, logger, writer,
          ngpus_per_node, args):
    if is_first_gpu(args, ngpus_per_node):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        column = [batch_time, data_time, losses, top1, top5]
        progress = ProgressMeter(logger,
                                 len(train_loader),
                                 column,
                                 prefix=f'Epoch: [{epoch}]')

    # switch to train mode
    model.train()
    if hasattr(model, 'module'):
        m = model.module
    else:
        m = model
    m.set_bin_activation()
    m.set_bin_weight()

    base_step = epoch * len(train_loader)
    optimizer.zero_grad()
    if is_first_gpu(args, ngpus_per_node):
        end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if is_first_gpu(args, ngpus_per_node):
            # measure data loading time
            data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        batchsize = torch.tensor(target.size(0)).cuda(args.gpu)
        if args.distributed:
            dist.barrier()
            dist.all_reduce(batchsize)

        optimizer.zero_grad()

        output, _ = model(images)
        loss = criterion(output, target)
        # do step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 *= target.size(0)
        acc5 *= target.size(0)
        if args.distributed:
            dist.barrier()
            dist.all_reduce(acc1)
            dist.all_reduce(acc5)
        acc1 /= batchsize
        acc5 /= batchsize
        if is_first_gpu(args, ngpus_per_node):
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], batchsize)
            top5.update(acc5[0], batchsize)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0 and i > 0:
                progress.display(i)
                if writer is not None:
                    writer.add_scalar('train/lr',
                                      optimizer.param_groups[0]['lr'],
                                      base_step + i)
                    writer.add_scalar('train/acc1', top1.avg, base_step + i)
                    writer.add_scalar('train/acc5', top5.avg, base_step + i)
                    writer.add_scalar('train/loss', loss.item(), base_step + i)
                # info_dict = {'train/lr': optimizer.param_groups[0]['lr'],
                #              'train/acc1': top1.avg,
                #              'train/acc5': top5.avg,
                #              'train/loss': loss_sum.item(),
                #              'train/loss_cls': loss_cls.item()}
                # wandb.log(info_dict)


def validate(val_loader, model, criterion, logger, writer, ngpus_per_node,
             args):
    if is_first_gpu(args, ngpus_per_node):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(logger,
                                 len(val_loader),
                                 [batch_time, losses, top1, top5],
                                 prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            batchsize = torch.tensor(target.size(0)).cuda(args.gpu)
            if args.distributed:
                dist.barrier()
                dist.all_reduce(batchsize)

            # compute output
            output, _ = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 *= target.size(0)
            acc5 *= target.size(0)
            if args.distributed:
                dist.barrier()
                dist.all_reduce(acc1)
                dist.all_reduce(acc5)
            acc1 /= batchsize
            acc5 /= batchsize
            if is_first_gpu(args, ngpus_per_node):
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], batchsize)
                top5.update(acc5[0], batchsize)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0 and i > 0:
                    progress.display(i)
        if is_first_gpu(args, ngpus_per_node):
            # TODO: this should also be done with the ProgressMeter
            logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))

    if is_first_gpu(args, ngpus_per_node):
        return top1.avg, top5.avg
    else:
        return -1, -1


def save_checkpoint(state, args, name='checkpoint.pth.tar'):
    filename = osp.join(args.logdir, name)
    torch.save(state, filename)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:

    def __init__(self, logger, num_batches, meters, prefix=''):
        self.logger = logger
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_lr_scheduler(optimizer, args):
    if args.lr_scheduler == 'CosineAnnealingLR':
        print('Use cosine scheduler')
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'StepLR':
        print('Use step scheduler, step size: {}, gamma: {}'.format(
            args.step_size, args.gamma))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        print('Use MultiStepLR scheduler, milestones: {}, gamma: {}'.format(
            args.milestones, args.gamma))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)
    else:
        raise NotImplementedError
    if args.warm_up:
        print('Use warm_up scheduler')
        lr_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.warmup_multiplier,
            total_epoch=args.warmup_epochs,
            after_scheduler=lr_scheduler)
        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()
    return lr_scheduler


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        num_classes = output.size(1) # Get the number of classes from the output tensor
        
        valid_topk_values = []
        for k_val_original in topk:
            if k_val_original <= num_classes:
                valid_topk_values.append(k_val_original)
        
        if not valid_topk_values: # If all k in topk were > num_classes
            valid_topk_values = [1] # Default to calculating at least top-1

        maxk_to_compute = max(valid_topk_values) 
        batch_size = target.size(0)

        _, pred = output.topk(maxk_to_compute, 1, True, True) # Uses adjusted maxk
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        computed_accuracies = {} 
        for k_computed in valid_topk_values:
            correct_k = correct[:k_computed].reshape(-1).float().sum(0, keepdim=True)
            computed_accuracies[k_computed] = correct_k.mul_(100.0 / batch_size)
        
        final_results = []
        for k_original_request in topk:
            if k_original_request in computed_accuracies:
                final_results.append(computed_accuracies[k_original_request])
            elif computed_accuracies: # If k_original > num_classes, use best available
                final_results.append(computed_accuracies[maxk_to_compute])
            else: 
                final_results.append(torch.tensor([0.0], device=output.device))
                
        return final_results


if __name__ == '__main__':
    main()
