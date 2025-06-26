# check_ops.py
import torch
import models 
import argparse 

if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='Check OPs for a model architecture')
    parser.add_argument('-a', '--arch', default='superbnn_wakevision_large', 
                        choices=model_names,
                        help='Model architecture to check')
    # VVVVVV ADD THIS ARGUMENT VVVVVV
    parser.add_argument('--img-size', type=int, default=128, # Default to 128 for WakeVision if not specified
                        help='Input image size (square) for the model, e.g., 32, 128, 224')
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    args = parser.parse_args()

    print(f"Checking OPs for architecture: {args.arch} with image size: {args.img_size}x{args.img_size}")
    
    model_constructor = models.__dict__[args.arch]
    model_arch_instance = None

    # Check if the constructor expects img_size (specifically for your _large models)
    # A more robust way would be to inspect the signature, but this works for now.
    if "wakevision_large" in args.arch or "cifar10_large" in args.arch or "imagenet" in args.arch.lower(): # Add other archs that take img_size
        model_arch_instance = model_constructor(img_size=args.img_size) 
    else: # For models like superbnn_cifar10 that might have img_size hardcoded in their wrapper
        try:
            model_arch_instance = model_constructor(img_size=args.img_size) # Try passing it anyway
        except TypeError: # If it doesn't accept img_size
            print(f"Note: {args.arch} constructor doesn't take img_size, using its internal default.")
            model_arch_instance = model_constructor()
    
    if hasattr(model_arch_instance, 'smallest_cand') and hasattr(model_arch_instance, 'biggest_cand'):
        # Ensure model is on CPU for get_ops if it creates tensors internally without device spec
        # model_arch_instance.cpu() # Might not be necessary if get_ops is device-agnostic

        smallest_ops_data = model_arch_instance.get_ops(model_arch_instance.smallest_cand.to('cpu')) # Move cand to cpu
        biggest_ops_data = model_arch_instance.get_ops(model_arch_instance.biggest_cand.to('cpu'))   # Move cand to cpu
        
        smallest_ops = smallest_ops_data[2] 
        biggest_ops = biggest_ops_data[2]
        
        print(f"Smallest candidate OPs: {smallest_ops:.2f} M")
        print(f"Biggest candidate OPs: {biggest_ops:.2f} M")
    else:
        print(f"Error: Model architecture '{args.arch}' instance does not have 'smallest_cand' or 'biggest_cand'.")