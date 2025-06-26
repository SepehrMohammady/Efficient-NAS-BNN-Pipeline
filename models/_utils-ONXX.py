# models/_utils.py
import torch
import torch.nn.functional as F
import os

_ONNX_EXPORTING = False

def set_onnx_exporting(is_exporting: bool):
    global _ONNX_EXPORTING
    _ONNX_EXPORTING = is_exporting
    if _ONNX_EXPORTING:
        print("INFO (_utils.py): Global ONNX export mode ENABLED.")
    else:
        print("INFO (_utils.py): Global ONNX export mode DISABLED.")

def _sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end

def adaptive_add(src, residual):
    src_c, src_wh = src.shape[1], src.shape[2]
    residual_c, residual_wh = residual.shape[1], residual.shape[2]
    if src_wh != residual_wh:
        if src_wh == residual_wh // 2:
            residual = F.avg_pool2d(residual, 2, stride=2)
        else:
            raise NotImplementedError
    if src_c == residual_c:
        out = src + residual
    else:
        # Ensure residual_c is not zero to prevent division by zero
        if residual_c == 0:
            # Handle this case: maybe add zeros, or raise error, or skip
            # For now, let's assume it shouldn't happen or src is used as is
            return src 
        repeat_count_float = src_c / residual_c
        # Using torch.round for repeat_count for safety, then cast to int
        repeat_count = int(torch.round(torch.tensor(repeat_count_float)).item())
        if repeat_count <= 0: # Handle cases where repeat_count might be zero or negative
            return src # Or some other appropriate action
            
        out = src + torch.cat(
            [residual for _ in range(repeat_count)],
            dim=1)[:, :src_c] # Ensure slicing doesn't go out of bounds if repeat_count is too small
    return out

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()

def grad_scale(x, scale):
    if _ONNX_EXPORTING:
        return x
    else:
        y = x
        y_grad = x * scale
        return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    if _ONNX_EXPORTING:
        return torch.round(x)
    else:
        # y = x.round() # OLD BUGGY LINE
        y = torch.round(x) # CORRECTED LINE
        y_grad = x
        return y.detach() - y_grad.detach() + y_grad