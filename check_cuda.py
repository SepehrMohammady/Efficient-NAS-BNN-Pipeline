import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version reported by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Try to set and use device 0
    try:
        torch.cuda.set_device(0) # Explicitly set to GPU 0
        print(f"Current GPU (after setting to 0): {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        print("Attempting a small model and tensor operation on CUDA:0...")
        # Simple model
        model_test = torch.nn.Linear(10,1).cuda(0) # Move model to GPU 0
        print(f"Test model device: {next(model_test.parameters()).device}")
        
        # Dummy tensor
        a = torch.randn(5, 10).cuda(0) # Move tensor to GPU 0
        print(f"Test tensor 'a' device: {a.device}")
        
        # Operation
        b = model_test(a)
        print(f"Test output 'b' device: {b.device}")
        print(f"Small CUDA op successful. Output sum: {b.sum()}")
        
    except Exception as e:
        print(f"Error during CUDA operations: {e}")
else:
    print("CUDA not available, cannot perform GPU tests.")

print(f"\nChecking environment variable CUDA_VISIBLE_DEVICES...")
import os
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_visible_devices:
    print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")
else:
    print("CUDA_VISIBLE_DEVICES is NOT set or is empty.")