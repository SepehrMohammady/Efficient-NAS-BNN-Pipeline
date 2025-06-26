# test.py
import argparse
import logging
import os
import os.path as osp
import warnings
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm # Ensure tqdm is imported

import models # Your models module
from utils import get_logger, tuple2cand # Assuming get_logger and tuple2cand are in utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('supernet', type=str, 
                    help='Path to the trained supernet checkpoint (.pth.tar)')
parser.add_argument('data', metavar='DIR', help='Path to dataset (e.g., ./data/WakeVision_Prepared_V3)')
parser.add_argument('checkpoint', # This is actually the search result file (info.pth.tar)
                    type=str,
                    metavar='PATH',
                    help='Path to searched checkpoint (info.pth.tar from search phase)')
parser.add_argument('logdir', metavar='DIR', 
                    help='Directory to save test logs and results')
parser.add_argument('-a', '--arch', metavar='ARCH', default='superbnn_wakevision_large',
                    choices=model_names,
                    help='Model architecture: ' + ' | '.join(model_names) +
                         ' (default: superbnn_cifar10)')
parser.add_argument('--dataset', type=str, default='WakeVision',
                    help='imagenet | cifar10 | WakeVision')
parser.add_argument('--img-size', type=int, default=None, # <<< ADDED IMG_SIZE ARGUMENT
                    help='Input image size (square). Required for datasets like WakeVision.')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='Number of data loading workers (default: 16, will be set to 0 for Windows single GPU)')
parser.add_argument('--ops', type=int, default=0, # Changed default, should be specified
                    help='OPs key from Pareto front in info.pth.tar to test')
parser.add_argument('--max-train-iters', type=int, default=10,
                    help='Number of iterations for BN calibration')
parser.add_argument('--train-batch-size', type=int, default=128, # For BN calibration
                    help='Batch size for BN calibration loader')
parser.add_argument('--test-batch-size', type=int, default=128, # For final test
                    help='Batch size for test data loader')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='Number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='Node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='Url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='Distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='Seed for initializing training.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training (not for single GPU typically)')


def is_first_gpu(args, ngpus_per_node):
    return not args.multiprocessing_distributed or \
           (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

def seed(seed_val=0): # Renamed seed to seed_val
    # import os # Global
    import random # Global
    import sys
    import numpy as np
    # import torch # Global
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed_val)
    random.seed(seed_val)

class DataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def main():
    args = parser.parse_args()
    # seed(args.seed) # Moved to main_worker
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == 'env://' and args.world_size == -1:
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ['WORLD_SIZE'])
        else: # Default for single process
            args.world_size = 1 
            
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    seed(args.seed) # Call seed within each worker process
    args.gpu = gpu
    logger = None 
    if args.gpu is not None:
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
        logger = get_logger(name='Test', log_file=osp.join(args.logdir, 'test.log'),
                            log_level=logging.INFO)
        logger.info(args)
    
    if is_first_gpu(args, ngpus_per_node) and logger:
        logger.info(f"=> creating model '{args.arch}'")

    searched = torch.load(args.checkpoint, map_location='cpu') # args.checkpoint is info.pth.tar
    if args.ops not in searched['pareto_global']:
        errmsg = f"OPs key {args.ops} not found in Pareto front of {args.checkpoint}. Available keys: {list(searched['pareto_global'].keys())}"
        if logger: logger.error(errmsg)
        else: print(f"ERROR: {errmsg}")
        return
        
    arch_tuple = searched['pareto_global'][args.ops]
    # Ensure arch_tuple is a tensor as expected by SuperBNN's sub_path handling if model created with it
    # tuple2cand converts the flat tuple from info.pth.tar into the N x 6 tensor
    arch_cand_tensor = tuple2cand(arch_tuple) 

    model = models.__dict__[args.arch](sub_path=arch_cand_tensor) 

    flops, bitops, total_flops = model.get_ops() # Uses model.sub_path_config_for_static_model which is arch_cand_tensor
    
    if is_first_gpu(args, ngpus_per_node) and logger:
        logger.info(f"=> Model for testing: '{args.arch}'") # Changed from creating model again
        logger.info(f"=> Search arch (from key {args.ops}): '{arch_tuple}'")
        logger.info('=> FLOPs {:.4f}M, BitOPs {:.4f}M, Total OPs (FLOPs + BitOPs/64) {:.4f}M'.format(
            flops, bitops, total_flops))
    
    if os.path.isfile(args.supernet):
        if is_first_gpu(args, ngpus_per_node) and logger:
            logger.info(f"=> Loading supernet checkpoint: '{args.supernet}'")
        supernet_checkpoint = torch.load(args.supernet, map_location='cpu')
        
        # Prepare state_dict for loading into the specific architecture (SuperBNN instance)
        # This typically involves loading the full supernet state_dict into the SuperBNN object,
        # as the sub-architecture uses shared weights from it.
        # The original script does: model.load_state_dict(state_dict_from_supernet)
        # This assumes the SuperBNN object (model) can handle this.
        supernet_state_dict_cleaned = {}
        for k, v in supernet_checkpoint['state_dict'].items():
            supernet_state_dict_cleaned[k.replace('module.', '')] = v
        model.load_state_dict(supernet_state_dict_cleaned, strict=False) # strict=False because model is a sub-arch
        
        if is_first_gpu(args, ngpus_per_node) and logger:
            logger.info(f"=> Loaded supernet checkpoint weights into model instance: '{args.supernet}'")
    else:
        if is_first_gpu(args, ngpus_per_node) and logger:
            logger.info(f"=> No supernet checkpoint found at '{args.supernet}'. Cannot proceed.")
        exit(0)

    # --- Dummy input creation for to_static ---
    dummy_input_size = 0
    if args.dataset == 'imagenet':
        dummy_input_size = getattr(args, 'img_size', 224)
    elif args.dataset == 'cifar10':
        dummy_input_size = 32
    elif args.dataset == 'WakeVision':
        if args.img_size is None:
            args.img_size = 128 # Default if not passed, but should be passed
            if logger and is_first_gpu(args, ngpus_per_node): logger.warning(f"--img-size not provided for WakeVision, defaulting to {args.img_size}.")
            elif is_first_gpu(args, ngpus_per_node): print(f"Warning: --img-size not provided for WakeVision, defaulting to {args.img_size}.")
        dummy_input_size = args.img_size
    else:
        err_msg_dataset = f"Unknown dataset '{args.dataset}' for dummy_input creation in test.py"
        if logger and is_first_gpu(args, ngpus_per_node): logger.error(err_msg_dataset)
        raise NotImplementedError(err_msg_dataset)
    
    dummy_input = torch.randn((1, 3, dummy_input_size, dummy_input_size))
    
    # --- GPU Setup for Model before to_static and evaluation ---
    # The model is moved to GPU first, then to_static is called.
    # to_static internally uses the device of the dummy_input if passed, or model's device.
    target_device = torch.device('cpu')
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        target_device = torch.device(f'cuda:{args.gpu}')
    elif torch.cuda.is_available() and not args.distributed and ngpus_per_node > 0 : # Default to DataParallel for multiple GPUs if not distributed and no specific GPU
        model = torch.nn.DataParallel(model).cuda()
        target_device = torch.device('cuda:0') # Assuming DataParallel uses cuda:0 primarily or dummy input on cuda:0

    dummy_input = dummy_input.to(target_device) # Move dummy input to the same device as model

    model.eval()
    # SuperBNN.to_static expects dummy_input and the sub_path_tuples (which is model.sub_path_config_for_static_model)
    # It uses self.sub_path_config_for_static_model if sub_path_tuples is None.
    # Since 'model' was initialized with sub_path=arch_cand_tensor, this is already set.
    model.to_static(dummy_input) # Call with dummy_input on correct device
    if logger and is_first_gpu(args, ngpus_per_node): logger.info("Model converted to static form.")

    # DataLoaders (train_loader for BN calib, val_loader for test)
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val') # This is the validation set for testing
    
    train_transform = None
    val_transform = None

    if args.dataset == 'imagenet':
        img_s_loader = getattr(args, 'img_size', 224)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        train_transform = transforms.Compose([transforms.RandomResizedCrop(img_s_loader), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(int(img_s_loader * 256 / 224)), transforms.CenterCrop(img_s_loader), transforms.ToTensor(), normalize])
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif args.dataset == 'WakeVision':
        img_s_loader = getattr(args, 'img_size', 128)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        train_transform = transforms.Compose([transforms.Resize((img_s_loader, img_s_loader)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize((img_s_loader, img_s_loader)), transforms.ToTensor(), normalize])
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported for data loading in test.py")

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=args.rank, num_replicas=args.world_size)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, rank=args.rank, num_replicas=args.world_size, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Adjust batch sizes for DDP if used
    actual_train_batch_size = args.train_batch_size
    actual_test_batch_size = args.test_batch_size
    actual_workers = args.workers
    if args.distributed and args.gpu is not None : # DDP per-GPU specific setup
        actual_train_batch_size = int(args.train_batch_size / ngpus_per_node)
        actual_test_batch_size = int(args.test_batch_size / ngpus_per_node)
        actual_workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=actual_train_batch_size, shuffle=(train_sampler is None),
        num_workers=actual_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=actual_test_batch_size, shuffle=False,
        num_workers=actual_workers, pin_memory=True, sampler=val_sampler)

    # BN Calibration
    if logger and is_first_gpu(args, ngpus_per_node): logger.info('Calibrating running stats for BN....')
    elif is_first_gpu(args, ngpus_per_node): print('Calibrating running stats for BN....')
    
    # Model should already be in eval() from to_static, but re-affirm for BN calib part
    model.eval() 
    for m_module in model.modules():
        if isinstance(m_module, nn.BatchNorm2d):
            # For BN calibration of a static model, we re-estimate running stats.
            # This means bn.training should be effectively True for stat collection,
            # but the model parameters (weights/biases) should not update.
            m_module.training = True 
            m_module.momentum = None  # Use cumulative moving average
            m_module.reset_running_stats()

    train_provider = DataIterator(train_loader) # For BN calib
    with torch.no_grad():
        # Use tqdm for progress bar if logger is not None (first GPU)
        bn_calib_iterable = tqdm.tqdm(range(args.max_train_iters), desc="BN Calibration", disable=not (is_first_gpu(args, ngpus_per_node)))
        for _ in bn_calib_iterable:
            images, _ = train_provider.next() # Target not used for BN calib forward pass
            if args.gpu is not None and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
            
            # The model's forward pass for SuperBNN (static) should just take images
            # as sub_path is already configured internally by to_static
            _ = model(images) 
            del images # Free memory

    # Evaluation
    # eval_device = next(model.parameters()).device if list(model.parameters()) else target_device
    eval_device = target_device # Use the device model was placed on

    top1_sum = torch.tensor([0.], device=eval_device)
    top5_sum = torch.tensor([0.], device=eval_device) # If you need top5
    total_samples = torch.tensor([0.], device=eval_device)

    if logger and is_first_gpu(args, ngpus_per_node): logger.info('Starting test....')
    elif is_first_gpu(args, ngpus_per_node): print('Starting test....')
    
    model.eval() # Ensure model is in eval mode for final testing

    val_loader_iterable = tqdm.tqdm(val_loader, desc="Testing", disable=not (is_first_gpu(args, ngpus_per_node)))
    for images, target_labels in val_loader_iterable: # Renamed target to target_labels
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target_labels = target_labels.cuda(args.gpu, non_blocking=True)
        
        current_batch_size = images.shape[0]
        output_logits, _ = model(images) # Model's forward call
        
        # accuracy function now returns a list of tensors
        acc_results = accuracy(output_logits, target_labels, topk=(1, 5)) 
        acc1 = acc_results[0]
        acc5 = acc_results[1] if len(acc_results) > 1 else acc_results[0] # Handle if only top1 returned

        top1_sum += acc1.item() * current_batch_size
        top5_sum += acc5.item() * current_batch_size
        total_samples += current_batch_size
        del images, target_labels, output_logits, acc1, acc5

    if args.distributed:
        dist.barrier()
        dist.all_reduce(top1_sum)
        dist.all_reduce(top5_sum)
        dist.all_reduce(total_samples)
    
    final_top1 = (top1_sum.item() / total_samples.item()) if total_samples.item() > 0 else 0.0
    final_top5 = (top5_sum.item() / total_samples.item()) if total_samples.item() > 0 else 0.0

    if is_first_gpu(args, ngpus_per_node) and logger:
        logger.info('Top-1 Accuracy: {:.2f}% Top-5 Accuracy: {:.2f}% on {} test images'.format(
            final_top1, final_top5, int(total_samples.item())))
    elif is_first_gpu(args, ngpus_per_node): # Fallback print
         print('Top-1 Accuracy: {:.2f}% Top-5 Accuracy: {:.2f}% on {} test images'.format(
            final_top1, final_top5, int(total_samples.item())))

def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        num_classes = output.size(1)
        
        valid_topk_values = []
        for k_val_original in topk:
            if k_val_original <= num_classes:
                valid_topk_values.append(k_val_original)
        if not valid_topk_values: valid_topk_values = [1]
            
        maxk_to_compute = max(valid_topk_values)
        batch_size = target.size(0)

        _, pred = output.topk(maxk_to_compute, 1, True, True)
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
            elif computed_accuracies:
                final_results.append(computed_accuracies[maxk_to_compute])
            else:
                final_results.append(torch.tensor([0.0], device=output.device))
        return final_results

if __name__ == '__main__':
    main()