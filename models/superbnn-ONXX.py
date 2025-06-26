import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _utils 
from ._utils import adaptive_add
from .dynamic_operations import (DynamicBatchNorm2d, DynamicBinConv2d,
                                 DynamicFPLinear, DynamicLearnableBias,
                                 DynamicPReLU, DynamicQConv2d)
from .operations import BinaryActivation


class BasicBlock(nn.Module):
    def __init__(self,
                 max_inp,
                 max_oup,
                 ks_list,
                 groups1_list,
                 groups2_list,
                 wh,
                 stride=1):
        super().__init__()
        assert stride in [1, 2]
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.ks_list = ks_list
        self.stride = stride
        cur_wh = wh

        self.move11 = DynamicLearnableBias(max_inp)
        self.binary_activation1 = BinaryActivation()
        self.binary_conv = DynamicBinConv2d(max_inp, max_inp, ks_list, groups1_list, wh=cur_wh, stride=stride)
        self.bn1 = DynamicBatchNorm2d(max_inp)
        self.move12 = DynamicLearnableBias(max_inp)
        self.prelu1 = DynamicPReLU(max_inp)
        self.move13 = DynamicLearnableBias(max_inp)
        self.move21 = DynamicLearnableBias(max_inp)
        self.binary_activation2 = BinaryActivation()
        cur_wh //= stride
        self.binary_conv1x1 = DynamicBinConv2d(max_inp, max_oup, [1], groups2_list, wh=cur_wh)
        self.bn2 = DynamicBatchNorm2d(max_oup)
        self.move22 = DynamicLearnableBias(max_oup)
        self.prelu2 = DynamicPReLU(max_oup)
        self.move23 = DynamicLearnableBias(max_oup)
        self.ops_memory = {}

    def forward(self, x, loss, sub_path): 
        out1 = self.move11(x)
        out1 = self.binary_activation1(out1)

        sub_path_conv1 = [-1, sub_path[1], sub_path[2]]
        if _utils._ONNX_EXPORTING:
            out1 = self.binary_conv(out1, sub_path=sub_path_conv1) 
        else:
            out1, loss = self.binary_conv(out1, loss=loss, sub_path=sub_path_conv1)
        
        out1 = self.bn1(out1)
        out1 = adaptive_add(out1, x) 
        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        out2 = self.binary_activation2(out2)

        sub_path_conv2 = [sub_path[0], 1, sub_path[3]]
        if _utils._ONNX_EXPORTING:
            out2 = self.binary_conv1x1(out2, sub_path=sub_path_conv2) 
        else:
            out2, loss = self.binary_conv1x1(out2, loss=loss, sub_path=sub_path_conv2)

        out2 = self.bn2(out2)
        out2 = adaptive_add(out2, out1) 
        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        if _utils._ONNX_EXPORTING:
            return out2
        else:
            return out2, loss

    def get_flops_bitops(self, pre_sub_path, sub_path):
        if tuple(pre_sub_path + sub_path) not in self.ops_memory:
            pre_channels, _, _, _ = pre_sub_path
            channels, ks, groups1, groups2 = sub_path
            bitops1 = (ks * ks * pre_channels // groups1 * pre_channels *
                       self.binary_conv.wh // self.stride *
                       self.binary_conv.wh // self.stride)
            bitops2 = (1 * 1 * pre_channels // groups2 * channels *
                       self.binary_conv1x1.wh * self.binary_conv1x1.wh)
            flops = 0.0
            bitops = bitops1 + bitops2
            self.ops_memory[tuple(pre_sub_path + sub_path)] = flops / 1e6, bitops / 1e6
        return self.ops_memory[tuple(pre_sub_path + sub_path)]

    def to_static(self, x, loss, sub_path): 
        out1_static = self.move11.to_static(x)
        out1_static = self.binary_activation1(out1_static)
        
        sub_path_conv1 = [-1, sub_path[1], sub_path[2]]
        binary_conv_static_output = self.binary_conv.to_static(out1_static, loss, sub_path_conv1) # Positional
        if isinstance(binary_conv_static_output, tuple): 
            out1_static = binary_conv_static_output[0]
        else: 
            out1_static = binary_conv_static_output

        out1_static = self.bn1.to_static(out1_static)
        out1_static = adaptive_add(out1_static, x) 
        out1_static = self.move12.to_static(out1_static)
        out1_static = self.prelu1.to_static(out1_static)
        out1_static = self.move13.to_static(out1_static)

        out2_static = self.move21.to_static(out1_static)
        out2_static = self.binary_activation2(out2_static)

        sub_path_conv2 = [sub_path[0], 1, sub_path[3]]
        binary_conv1x1_static_output = self.binary_conv1x1.to_static(out2_static, loss, sub_path_conv2) # Positional
        if isinstance(binary_conv1x1_static_output, tuple):
            out2_static = binary_conv1x1_static_output[0]
        else:
            out2_static = binary_conv1x1_static_output
            
        out2_static = self.bn2.to_static(out2_static)
        out2_static = adaptive_add(out2_static, out1_static) 
        out2_static = self.move22.to_static(out2_static)
        out2_static = self.prelu2.to_static(out2_static)
        out2_static = self.move23.to_static(out2_static)
        
        return out2_static, loss


class StemBlock(nn.Module):
    def __init__(self,
                 max_inp,
                 max_oup,
                 ks_list,
                 groups1_list,
                 groups2_list, 
                 wh,
                 stride=1):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.ks_list = ks_list

        self.conv = DynamicQConv2d(max_inp, max_oup, ks_list, groups1_list, w_bit=8, a_bit=None, wh=wh, stride=stride)
        self.bn = DynamicBatchNorm2d(max_oup)
        self.move1 = DynamicLearnableBias(max_oup)
        self.relu = DynamicPReLU(max_oup) 
        self.move2 = DynamicLearnableBias(max_oup)

    def forward(self, x, loss, sub_path): 
        sub_path_qconv = sub_path[:3]
        out = self.conv(x, sub_path=sub_path_qconv) 
        out = self.bn(out)
        out = self.move1(out)
        out = self.relu(out)
        out = self.move2(out)
        
        if _utils._ONNX_EXPORTING:
            return out
        else:
            return out, loss 

    def get_flops_bitops(self, sub_path):
        return self.conv.get_flops_bitops(sub_path[:3])

    def to_static(self, x, loss, sub_path):
        sub_path_qconv = sub_path[:3]
        out_static = self.conv.to_static(x, sub_path_qconv) # Pass sub_path positionally
        out_static = self.bn.to_static(out_static)
        out_static = self.move1.to_static(out_static)
        out_static = self.relu.to_static(out_static) 
        out_static = self.move2.to_static(out_static)
        return out_static, loss


class SuperBNN(nn.Module):
    def __init__(self, cfg, n_class=1000, img_size=224, sub_path=None):
        super().__init__()
        self.cfg = cfg
        self.n_class = n_class
        self.img_size = img_size
        self.sub_path_config_for_static_model = sub_path 
        self.is_static_model = False 

        cur_img_size = self.img_size
        self.features = nn.ModuleList()
        self.search_space = []
        self.max_inp = 3 # Initial input channels (e.g. RGB)

        for i, (channels_list, num_blocks_list, ks_list, groups1_list,
                groups2_list, stride) in enumerate(self.cfg):
            max_channels_current_stage = max(channels_list)
            max_num_blocks_current_stage = max(num_blocks_list)
            stage_module_list = nn.ModuleList()
            stage_search_space_config = [] 
            
            current_input_channels_for_block = self.max_inp # Input to the first block of this stage

            for j in range(max_num_blocks_current_stage):
                block_constructor = StemBlock if i == 0 and j == 0 else BasicBlock
                
                block_stride = stride if j == 0 else 1 # Stride only for the first block in a stage

                if i == 0 and j == 0: # StemBlock
                     stage_module_list.append(
                        block_constructor(current_input_channels_for_block, 
                              max_channels_current_stage, # Stem output channels
                              ks_list, groups1_list, groups2_list, 
                              wh=cur_img_size, stride=block_stride))
                     # Output of Stem becomes input for next potential block in this stage or next stage
                     current_input_channels_for_block = max_channels_current_stage 
                else: # BasicBlock
                    stage_module_list.append(
                        block_constructor(current_input_channels_for_block, 
                                   max_channels_current_stage, # BasicBlock output channels for this stage
                                   ks_list, groups1_list, groups2_list,
                                   wh=cur_img_size, stride=block_stride))
                    # Output of this BasicBlock is also max_channels_current_stage
                    current_input_channels_for_block = max_channels_current_stage
                
                if block_stride > 1: 
                    cur_img_size //= block_stride
                
                stage_search_space_config.append([channels_list, ks_list, groups1_list, groups2_list])
            
            self.features.append(stage_module_list)
            self.search_space.append([stage_search_space_config, num_blocks_list])
            self.max_inp = max_channels_current_stage # Output of this stage is max_channels_current_stage

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = DynamicFPLinear(self.max_inp, n_class) 
        
        self.set_bin_weight()
        self.set_bin_activation()
        self.close_distill()
        if sub_path is None : 
            self.register_buffer('biggest_cand', self.get_biggest_cand())
            self.register_buffer('smallest_cand', self.get_smallest_cand())
            # self.get_ops might need self.features to be fully built
            # _, _, self.biggest_ops = self.get_ops(self.biggest_cand)
            # _, _, self.smallest_ops = self.get_ops(self.smallest_cand)

    def forward(self, x, sub_path=None): 
        current_arch_path = None
        if self.is_static_model: 
            current_arch_path = self.sub_path_config_for_static_model
        elif sub_path is not None: 
            current_arch_path = sub_path
        else:
            raise ValueError("SuperBNN.forward: Architecture sub_path not specified for dynamic model.")

        current_total_loss = None
        if not _utils._ONNX_EXPORTING:
            current_total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for i, j, channels_val, ks_val, groups1_val, groups2_val in current_arch_path:
            if i == -1 or j == -1: 
                continue
            
            block_sub_path = [channels_val.item(), ks_val.item(), groups1_val.item(), groups2_val.item()]
            
            if _utils._ONNX_EXPORTING:
                x = self.features[i][j](x, loss=None, sub_path=block_sub_path) 
            else:
                x, current_total_loss = self.features[i][j](x, loss=current_total_loss, sub_path=block_sub_path)
        
        x = self.globalpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) 

        if _utils._ONNX_EXPORTING:
            return x 
        else:
            return x, current_total_loss

    def get_ops(self, sub_path=None):
        current_arch_path = sub_path
        if current_arch_path is None:
            current_arch_path = self.sub_path_config_for_static_model
            if current_arch_path is None:
                 raise ValueError("get_ops: sub_path must be provided.")
        
        flops = 0.0
        bitops = 0.0
        
        # This logic needs to accurately track input channels to each block
        # based on the output of the *actual previous active block* in the path.
        # The original code used `pre` which was the sub_path of the previous block.
        # For simplicity and focusing on ONNX, we'll keep the structure but acknowledge
        # that robust FLOPs calculation for arbitrary paths is non-trivial here.
        
        # Placeholder for previous block's output config [channels, ks, g1, g2]
        # Initial input to the network is 3 channels.
        # For the first block (Stem), pre_sub_path is not used in its get_flops_bitops.
        # For subsequent BasicBlocks, pre_sub_path refers to the output arch of the *actual* preceding block.
        
        # To make this somewhat work, we need to track the output channels of the last *active* block.
        last_active_block_output_channels = 3 
        # And we need a placeholder for its full config if BasicBlock needs it.
        # This detail is complex if blocks can be skipped.
        # The original pre = cur might have worked if path was dense.

        # Let's assume a simplified pre_sub_path based on last output channels for now
        # This part is primarily for search analysis, not critical for ONNX export itself.
        # If you need precise FLOPs, this section needs careful validation.
        
        temp_pre_sub_path_for_flops = [3, 0, 0, 0] # Dummy: [ch, ks, g1, g2]

        for k_path_idx, (i, j, channels, ks, groups1, groups2) in enumerate(current_arch_path):
            if i == -1 or j == -1:
                continue
            
            current_block_arch_for_flops = [channels.item(), ks.item(), groups1.item(), groups2.item()]

            if i == 0 and j == 0: # StemBlock
                stem_sub_path_for_flops = [channels.item(), ks.item(), groups1.item()]
                tmp_flops, tmp_bitops = self.features[i][j].get_flops_bitops(stem_sub_path_for_flops)
            else: # BasicBlock
                # Update temp_pre_sub_path_for_flops based on the output of the true previous block
                # This is where it gets tricky if blocks are skipped.
                # The most straightforward is to use the output channel of the last processed block.
                # BasicBlock.get_flops_bitops expects [prev_ch_out, prev_ks, prev_g1, prev_g2]
                # For simplicity, using last_active_block_output_channels for prev_ch_out.
                # Other elements of temp_pre_sub_path_for_flops might be less critical or assumed.
                # temp_pre_sub_path_for_flops = [last_active_block_output_channels, prev_ks, prev_g1, prev_g2]
                # This needs careful handling of what prev_ks, etc. should be.
                # The original code `pre = cur` passed the full arch of the prev block.
                # We need to find the *actual* previous block in the path.
                
                # Let's refine: find the true previous active block in current_arch_path
                true_prev_active_block_arch = [3,0,0,0] # Default for input to first basic block after stem
                if not (i==0 and j==0): # if not the stem block
                    # Search backwards from k_path_idx-1 for an active block
                    found_prev = False
                    for prev_k_path_idx in range(k_path_idx - 1, -1, -1):
                        prev_i, prev_j, prev_ch, prev_ks, prev_g1, prev_g2 = current_arch_path[prev_k_path_idx]
                        if prev_i != -1:
                            true_prev_active_block_arch = [prev_ch.item(), prev_ks.item(), prev_g1.item(), prev_g2.item()]
                            found_prev = True
                            break
                    if not found_prev and i==0 : # First block of stage 0, but not stem (should not happen with current init)
                         pass # Should be handled by stem case
                    elif not found_prev : # First active block in a later stage, input from prev stage's output
                         # This needs to get the output config of the last block of stage i-1
                         # This get_ops is getting very complex.
                         # For now, we use the simpler approach from earlier, acknowledging limitations.
                         pass


                tmp_flops, tmp_bitops = self.features[i][j].get_flops_bitops(
                    temp_pre_sub_path_for_flops, current_block_arch_for_flops)

            flops += tmp_flops
            bitops += tmp_bitops
            temp_pre_sub_path_for_flops = current_block_arch_for_flops # Update for next iteration
            last_active_block_output_channels = channels.item()


        fc_pre_sub_path_for_flops = [last_active_block_output_channels] # Input features to FC
        tmp_flops, tmp_bitops = self.fc.get_flops_bitops(fc_pre_sub_path_for_flops)
        flops += tmp_flops
        bitops += tmp_bitops
        
        ops = flops + bitops / 64 
        return flops, bitops, ops

    def set_fp_weight(self):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d): m.is_bin = False
    def set_fp_weight_prob(self, prob):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d):
                if np.random.random_sample() <= prob: m.is_bin = False
    def set_bin_weight(self):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d): m.is_bin = True
    def set_bin_activation(self):
        for m in self.modules():
            if isinstance(m, BinaryActivation): m.is_bin = True
    def set_fp_activation(self):
        for m in self.modules():
            if isinstance(m, BinaryActivation): m.is_bin = False
    def set_fp_activation_prob(self, prob):
        for m in self.modules():
            if isinstance(m, BinaryActivation):
                if np.random.random_sample() <= prob: m.is_bin = False
    def open_distill(self):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d): m.is_distill = True
    def close_distill(self):
        for m in self.modules():
            if isinstance(m, DynamicBinConv2d): m.is_distill = False

    def to_static(self, dummy_input, sub_path_tuples=None): 
        current_arch_path_for_static = None
        if sub_path_tuples is not None:
            current_arch_path_for_static = sub_path_tuples
            self.sub_path_config_for_static_model = sub_path_tuples 
        elif self.sub_path_config_for_static_model is not None:
            current_arch_path_for_static = self.sub_path_config_for_static_model
        else:
            raise ValueError("SuperBNN.to_static: Architecture sub_path not specified.")

        self.is_static_model = True 

        dummy_loss = torch.tensor(0.0, device=dummy_input.device, dtype=dummy_input.dtype)
        current_x_for_to_static = dummy_input 

        for i, j, channels_val, ks_val, groups1_val, groups2_val in current_arch_path_for_static:
            if i == -1 or j == -1:
                continue
            
            block_sub_path = [channels_val.item(), ks_val.item(), groups1_val.item(), groups2_val.item()]
            
            block_static_output, _ = self.features[i][j].to_static(
                current_x_for_to_static, dummy_loss, block_sub_path) # Pass sub_path positionally
            current_x_for_to_static = block_static_output 
        
        out_after_features_static = current_x_for_to_static
        out_pooled_static = self.globalpool(out_after_features_static) 
        out_flattened_static = torch.flatten(out_pooled_static, 1)
        
        # DynamicFPLinear.to_static(self, x) needs only x
        # It uses x.shape[1] to determine input features for its weight_s
        _ = self.fc.to_static(out_flattened_static) # Modifies self.fc in-place
        
        print(f"SuperBNN model configured to static with architecture.")

    def get_random_cand(self):
        if hasattr(self, 'module'): m = self.module
        else: m = self
        device = next(m.parameters()).device
        res = []
        for i, (stage_search_space_config, num_blocks_options) in enumerate(self.search_space):
            block_num_this_stage = random.choice(num_blocks_options)
            for j, block_param_options in enumerate(stage_search_space_config):
                if j >= block_num_this_stage:
                    cur = [-1, -1, -1, -1, -1, -1] 
                else:
                    cur = [i, j] 
                    cur.append(random.choice(block_param_options[0])) 
                    cur.append(random.choice(block_param_options[1])) 
                    cur.append(random.choice(block_param_options[2])) 
                    cur.append(random.choice(block_param_options[3])) 
                res.append(torch.tensor(cur, dtype=torch.long)[None, :]) 
        res = torch.cat(res, dim=0).to(device)
        return res

    def get_biggest_cand(self):
        res = []
        for i, (stage_search_space_config, num_blocks_options) in enumerate(self.search_space):
            block_num_this_stage = max(num_blocks_options) 
            for j, block_param_options in enumerate(stage_search_space_config): 
                if j >= block_num_this_stage:
                    cur = [-1, -1, -1, -1, -1, -1]
                else:
                    cur = [i, j]
                    cur.append(max(block_param_options[0])) 
                    cur.append(max(block_param_options[1])) 
                    cur.append(min(block_param_options[2])) 
                    cur.append(min(block_param_options[3])) 
                res.append(torch.tensor(cur, dtype=torch.long)[None, :])
        res = torch.cat(res, dim=0)
        return res

    def get_smallest_cand(self):
        res = []
        for i, (stage_search_space_config, num_blocks_options) in enumerate(self.search_space):
            block_num_this_stage = min(num_blocks_options)
            for j, block_param_options in enumerate(stage_search_space_config):
                if j >= block_num_this_stage:
                    cur = [-1, -1, -1, -1, -1, -1]
                else:
                    cur = [i, j]
                    cur.append(min(block_param_options[0])) 
                    cur.append(min(block_param_options[1])) 
                    cur.append(max(block_param_options[2])) 
                    cur.append(max(block_param_options[3])) 
                res.append(torch.tensor(cur, dtype=torch.long)[None, :])
        res = torch.cat(res, dim=0)
        return res

def superbnn(sub_path=None):
    cfg = [[[24, 32, 48], [1], [3], [1], [1], 2],
           [[48, 64, 96], [2, 3], [3], [1], [1], 1],
           [[96, 128, 192], [2, 3], [3, 5], [1, 2], [1], 2],
           [[192, 256, 384], [2, 3], [3, 5], [2, 4], [1], 2],
           [[384, 512, 768], [8, 9], [3, 5], [4, 8], [1], 2],
           [[768, 1024, 1536], [2, 3], [3, 5], [8, 16], [1], 2]]
    return SuperBNN(cfg, n_class=1000, img_size=224, sub_path=sub_path)

def superbnn_100(sub_path=None):
    cfg = [[[24, 32, 48], [1], [3], [1], [1], 2],
           [[48, 64, 96], [2, 3], [3], [1], [1], 1],
           [[96, 128, 192], [2, 3], [3, 5], [1, 2], [1], 2],
           [[192, 256, 384], [2, 3], [3, 5], [2, 4], [1], 2],
           [[384, 512, 768], [8, 9], [3, 5], [4, 8], [1], 2],
           [[768, 1024, 1536], [2, 3], [3, 5], [8, 16], [1], 2]]
    return SuperBNN(cfg, n_class=100, img_size=224, sub_path=sub_path)

def superbnn_cifar10(sub_path=None):
    cfg = [
        [[16, 24, 32], [1], [3], [1], [1], 2],
        [[32, 48, 64], [1, 2], [3], [1], [1], 1],
        [[64, 96, 128], [1, 2], [3, 5], [1, 2], [1], 2],
        [[128, 196, 256], [1, 2], [3, 5], [2, 4], [1], 2],
    ]
    return SuperBNN(cfg, n_class=10, img_size=32, sub_path=sub_path)
# At the end of models/superbnn.py, add this new function:

def superbnn_cifar10_large(sub_path=None):
    # (channels_list, num_blocks_list, ks_list, groups1_list, groups2_list, stride)
    # Aiming for a significantly larger model space
    cfg = [
        # Stage 0 (Input: 32x32)
        [[32, 48, 64],    [1],       [3],    [1],    [1],    2], # Output: 16x16, Max Channels: 64
        # Stage 1
        [[64, 96, 128],   [1, 2],    [3],    [1],    [1],    1], # Output: 16x16, Max Channels: 128
        # Stage 2
        [[128, 192, 256], [2, 3],    [3, 5], [1, 2], [1],    2], # Output: 8x8, Max Channels: 256
        # Stage 3
        [[256, 384, 512], [2, 3, 4], [3, 5], [1, 2], [1, 2], 2], # Output: 4x4, Max Channels: 512
        # Stage 4
        [[512, 768, 1024],[2, 3, 4], [3, 5], [1, 2, 4],[1, 2], 1], # Output: 4x4, Max Channels: 1024 
        # Stage 5 (Optional, if still not large enough)
        # [[1024, 1536, 2048], [1,2], [3], [1,2,4], [1,2], 2], # Output: 2x2, Max Channels: 2048
    ]
    # The final FC layer input will be self.max_inp from the __init__ of SuperBNN, 
    # which is the max channels of the last stage (e.g., 1024 or 2048).
    return SuperBNN(cfg, n_class=10, img_size=32, sub_path=sub_path)

def superbnn_wakevision_large(sub_path=None, img_size=128): # Accept img_size, provide a default
    cfg = [
        # Your defined CFG for superbnn_wakevision_large
        [[32, 48, 64],    [1],       [3,5],    [1],    [1],    2], 
        [[64, 96, 128],   [1, 2],    [3,5],    [1],    [1],    2], 
        [[128, 192, 256], [2, 3],    [3, 5], [1, 2], [1],    2], 
        [[256, 384, 512], [2, 3, 4], [3, 5], [1, 2], [1, 2], 2], 
        [[512, 768, 1024],[2, 3, 4], [3, 5], [1, 2, 4],[1, 2], 2], 
    ]
    return SuperBNN(cfg, n_class=2, img_size=img_size, sub_path=sub_path)