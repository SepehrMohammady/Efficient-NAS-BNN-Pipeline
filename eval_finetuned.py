# eval_finetuned.py

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse
import models # Your models folder
from utils import tuple2cand, get_logger # Assuming get_logger and tuple2cand are in utils
import re 
import logging # Added import

def accuracy_calc(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k_val in topk:
            correct_k = correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name; self.fmt = fmt; self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a Fine-tuned NAS-BNN Model')
    parser.add_argument('finetuned_checkpoint', help='Path to the fine-tuned model checkpoint .pth.tar file')
    parser.add_argument('arch_key', type=int, help='OPs bucket key used to identify arch in search_info_file')
    parser.add_argument('search_info_file', help='Path to info.pth.tar from the search')
    parser.add_argument('--arch_name', default='superbnn_wakevision_large', help='Model architecture name in models.__dict__')
    parser.add_argument('--dataset_name', default='cifar10', help='Dataset name (cifar10 or imagenet for dummy input size)')
    parser.add_argument('--data_dir', default='./data/CIFAR10', help='Path to CIFAR10 dataset (or ImageNet if specified)')
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use, None for CPU")
    args = parser.parse_args()

    # Initialize a simple logger for this script
    log_dir_eval = os.path.dirname(args.finetuned_checkpoint) 
    eval_log_file = os.path.join(log_dir_eval, f"eval_finetuned_key{args.arch_key}.log")
    # Ensure log_dir_eval exists if it's different from script's current dir (e.g. a subfolder)
    if not os.path.exists(log_dir_eval) and log_dir_eval : # Check if log_dir_eval is not empty string
        os.makedirs(log_dir_eval, exist_ok=True)

    logger = get_logger(name=f'EvalFinetuned_Key{args.arch_key}', log_file=eval_log_file, log_level=logging.INFO, file_mode='a') # Use 'a' to append
    logger.info("--- Evaluation of Fine-tuned Model Started ---")
    logger.info(f"Arguments: {args}")

    # 1. Load architecture definition from search_info_file
    logger.info(f"Loading architecture definition from: {args.search_info_file}")
    try:
        search_results = torch.load(args.search_info_file, map_location='cpu')
        if args.arch_key not in search_results.get('pareto_global', {}):
            logger.error(f"ERROR: Architecture key {args.arch_key} not found in pareto_global of {args.search_info_file}")
            logger.error(f"Available keys: {list(search_results.get('pareto_global', {}).keys())}")
            return
        arch_tuple_stored = search_results['pareto_global'][args.arch_key]
        arch_cand_tensor = tuple2cand(arch_tuple_stored) 
        logger.info(f"Loaded architecture for key {args.arch_key}: {arch_tuple_stored}")
    except Exception as e:
        logger.error(f"Error loading or parsing search_info_file: {e}")
        return

    # 2. Instantiate the model
    logger.info(f"Instantiating model: {args.arch_name}")
    try:
        model = models.__dict__[args.arch_name](sub_path=arch_cand_tensor)
    except Exception as e:
        logger.error(f"Error instantiating model {args.arch_name}: {e}")
        return

    # 3. Determine device and convert model to static form
    logger.info("Preparing model for static evaluation...")
    device_for_model = torch.device(f"cuda:{args.gpu}") if args.gpu is not None and torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Target device for model and evaluation: {device_for_model}")

    # Move the entire model to the target device FIRST.
    model.to(device_for_model)
    model_params_on_device = "N/A"
    try: model_params_on_device = str(next(model.parameters()).device)
    except StopIteration: pass # Model has no parameters
    logger.info(f"Model (before to_static) moved to device: {model_params_on_device}")
    
    # Create dummy_input on the SAME device as the model.
    if args.dataset_name == 'cifar10':
        dummy_input_shape = (1, 3, 32, 32) 
    elif args.dataset_name == 'imagenet':
        dummy_input_shape = (1, 3, 224, 224)
    else:
        logger.error(f"Unknown dataset_name '{args.dataset_name}' for dummy input shape.")
        return
    dummy_input = torch.randn(dummy_input_shape, device=device_for_model)
    logger.info(f"Dummy input created on device: {dummy_input.device}")

    model.eval() # Set to eval mode first
    try:
        logger.info("Calling model.to_static()...")
        model.to_static(dummy_input) 
        logger.info("Model converted to static form.")
    except Exception as e:
        logger.error(f"Error during model.to_static(): {e}")
        logger.exception("Detailed traceback for to_static error:")
        return

    # 4. Load fine-tuned weights
    logger.info(f"Loading fine-tuned model weights from: {args.finetuned_checkpoint}")
    if not os.path.isfile(args.finetuned_checkpoint):
        logger.error(f"Fine-tuned checkpoint file not found: {args.finetuned_checkpoint}")
        return
    try:
        checkpoint = torch.load(args.finetuned_checkpoint, map_location='cpu') 
        state_dict = checkpoint['state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=True) 
        logger.info("Fine-tuned model weights loaded successfully (strict=True).")
    except RuntimeError as e:
        logger.warning(f"RuntimeError loading state_dict with strict=True: {e}")
        logger.warning("Attempting to load with strict=False...")
        try:
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("Fine-tuned model weights loaded successfully (strict=False).")
        except Exception as e_false:
            logger.error(f"Failed to load state_dict even with strict=False: {e_false}")
            return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading the checkpoint: {e}")
        return

    # Ensure model is on the correct device (should already be, but good check)
    model.to(device_for_model)
    final_model_device = "N/A"
    try: final_model_device = str(next(model.parameters()).device)
    except StopIteration: pass
    logger.info(f"Model confirmed on device: {final_model_device} for evaluation.")
    model.eval()

    # 5. Prepare DataLoader for the TEST set
    testdir = os.path.join(args.data_dir, 'test') 
    if not os.path.exists(testdir):
        logger.error(f"Test data directory not found: {testdir}. Please run prepare_cifar10.py or check path.")
        return
        
    normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                     std=[0.24703233, 0.24348505, 0.26158768]) 
    try:
        test_dataset = datasets.ImageFolder(
            testdir,
            transforms.Compose([transforms.ToTensor(), normalize]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True if args.gpu is not None and torch.cuda.is_available() else False)
        logger.info(f"Test DataLoader created for directory: {testdir} with {len(test_dataset)} images.")
    except Exception as e:
        logger.error(f"Error creating Test DataLoader: {e}")
        return

    # 6. Perform evaluation
    top1_avg_meter = AverageMeter('Acc@1', ':6.2f') 
    top5_avg_meter = AverageMeter('Acc@5', ':6.2f')

    logger.info("Starting evaluation on the TEST set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device_for_model, non_blocking=True)
            labels = labels.to(device_for_model, non_blocking=True)
            
            outputs, _ = model(images, model.sub_path) 
            
            acc1, acc5 = accuracy_calc(outputs, labels, topk=(1, 5)) 
            top1_avg_meter.update(acc1.item(), images.size(0))
            top5_avg_meter.update(acc5.item(), images.size(0))

    logger.info(f"--- TEST SET RESULTS for Key {args.arch_key} ---")
    logger.info(f'Top-1 Accuracy: {top1_avg_meter.avg:.2f}%')
    logger.info(f'Top-5 Accuracy: {top5_avg_meter.avg:.2f}%')
    logger.info(f"Evaluated on {len(test_dataset)} test images.")
    logger.info("--- Evaluation Complete ---")

if __name__ == '__main__':
    main()