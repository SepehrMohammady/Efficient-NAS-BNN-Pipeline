# In models/dynamic_operations.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the _utils module to access its global _ONNX_EXPORTING flag
from . import _utils # This makes _utils._ONNX_EXPORTING accessible
# And import specific functions you need directly if you prefer
from ._utils import _sub_filter_start_end, grad_scale, lp_loss, round_pass
from .operations import SignFuncW 

from ._utils import _ONNX_EXPORTING, round_pass, grad_scale # ... other needed utils

from . import _utils # Import the module itself
from ._utils import round_pass, grad_scale # Import specific functions if you still want to

class DynamicLearnableBias(nn.Module):
    def __init__(self, max_channels_for_dynamic_supernet_bias):
        super().__init__()
        # This 'bias' is primarily for the dynamic supernet phase if used before 'to_static'.
        # Its size is based on the max possible channels the supernet layer is designed for.
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, max_channels_for_dynamic_supernet_bias, 1, 1)))
        self.static_mode_active = False # Flag to indicate if to_static has created bias_s
        # self.bias_s (the static parameter) will be created in to_static.

    def forward(self, x):
        if self.static_mode_active and hasattr(self, 'bias_s'):
            # In static mode (after to_static has run), use the specific bias_s.
            # self.bias_s was sized based on the 'x' passed to 'to_static'.
            # The 'x' here should have the same number of channels.
            out = x + self.bias_s.expand_as(x)
        else:
            # Dynamic mode (supernet training/evaluation before to_static is called)
            feature_dim = x.size(1)
            # Slice the 'dynamic' self.bias parameter.
            # This requires x.size(1) <= self.bias.size(1) (max_channels_for_dynamic_supernet_bias).
            out = x + self.bias[:, :feature_dim].expand_as(x)
        return out

    def to_static(self, x): # x is the input tensor for this specific static configuration
        self.static_mode_active = True
        feature_dim = x.size(1) # Actual input channels for this static instance.

        # For the static model, bias_s should be sized purely by feature_dim.
        # The actual trained values for bias_s will be loaded from the finetuned checkpoint.
        # Initialize with zeros; checkpoint will overwrite.
        static_bias_s_data = torch.zeros(1, feature_dim, 1, 1, device=x.device, dtype=x.dtype)

        # Ensure bias_s is correctly registered or re-registered with the new size.
        if hasattr(self, 'bias_s'):
            # If to_static could be called multiple times with different shapes (not ideal),
            # properly handle re-registration.
            if self._parameters['bias_s'].shape[1] != feature_dim:
                del self._parameters['bias_s'] # Remove old one if size differs
                self.register_parameter('bias_s', nn.Parameter(static_bias_s_data))
            else:
                # If size matches, could just update data, but re-registering is cleaner if new data.
                # For zero init, just ensure it's there.
                pass # Parameter already exists with correct shape
        else:
            self.register_parameter('bias_s', nn.Parameter(static_bias_s_data))
        
        # Original dynamic 'bias' is no longer needed for gradient in static model.
        if hasattr(self, 'bias'):
            self.bias.requires_grad = False
            # Optionally, one might `del self.bias` if it's certain it won't be used again
            # and to ensure state_dict loading doesn't see unexpected keys.
            # However, if train_single.py loads a supernet checkpoint first, self.bias might be needed.
            # For ONNX export from a finetuned static model, self.bias is not the target.

        # This forward call is part of the to_static trace for this layer.
        # It will use the self.bias_s that was just defined/checked.
        return self.forward(x)


class DynamicQConv2d(nn.Module):
    def __init__(self,
                 max_inp,
                 max_oup,
                 ks_list,
                 groups_list,
                 w_bit,
                 a_bit,
                 wh,
                 stride=1,
                 dilation=1,
                 channel_wise=True):
        super().__init__()
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.ks_list = list(set(ks_list))
        self.ks_list.sort()
        self.max_ks = max(ks_list)
        self.groups_list = groups_list
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.wh = wh
        self.stride = stride
        self.dilation = dilation
        shape = (self.max_oup, self.max_inp // min(groups_list), self.max_ks,
                 self.max_ks)
        self.register_parameter('weight', nn.Parameter(torch.rand(shape)))
        if w_bit != 32:
            if channel_wise:
                self.scale_channel = max_oup
            else:
                self.scale_channel = 1
            self.register_parameter(
                'w_scale', nn.Parameter(torch.ones(self.scale_channel)))
            self.register_buffer('w_init_state', torch.zeros(1))
            self.w_min_q = -1 * 2**(w_bit - 1)
            self.w_max_q = 2**(w_bit - 1) - 1

        if a_bit != 32 and a_bit is not None:
            self.register_parameter('a_scale', nn.Parameter(torch.ones(1)))
            self.register_parameter('a_zero_point',
                                    nn.Parameter(torch.ones(1)))
            self.register_buffer('a_init_state', torch.zeros(1))
            self.a_min_q = 0
            self.a_max_q = 2**a_bit - 1

        if len(self.ks_list) > 1:
            scale_params = {}
            for i in range(len(self.ks_list) - 1):
                ks_small = self.ks_list[i]
                ks_larger = self.ks_list[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                scale_params['matrix_%s' % param_name] = nn.Parameter(
                    torch.eye(ks_small**2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)
        self.eps = torch.finfo(torch.float32).eps
        self.ops_memory = {}
        self.reset_parameters()
        self.static = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def get_active_filter(self, weight, inp, oup, ks, groups):
        start, end = _sub_filter_start_end(self.max_ks, ks)
        filters = weight[:oup, :inp, start:end, start:end]
        if ks < self.max_ks:
            start_filter = weight[:oup, :inp, :, :]
            for i in range(len(self.ks_list) - 1, 0, -1):
                src_ks = self.ks_list[i]
                if src_ks <= ks:
                    break
                target_ks = self.ks_list[i - 1]
                start_idx, end_idx = _sub_filter_start_end(src_ks, target_ks) # Renamed start, end
                _input_filter = start_filter[:, :, start_idx:end_idx, start_idx:end_idx]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0),
                                                   _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    getattr(self, 'matrix_%dto%d' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0),
                                                   filters.size(1),
                                                   target_ks**2)
                _input_filter = _input_filter.view(filters.size(0),
                                                   filters.size(1), target_ks,
                                                   target_ks)
                start_filter = _input_filter
            filters = start_filter
        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_inp = inp // groups
        sub_ratio = filters.size(1) // sub_inp
        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start_val = part_id * sub_inp
            filter_crops.append(sub_filter[:, start_val:start_val + sub_inp, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def forward(self, x, sub_path):
        oup_cfg, ks_cfg, groups_cfg = sub_path
        inp = x.shape[1]
        oup_actual = inp if oup_cfg == -1 else oup_cfg

        if self.static:
            weight_to_use = self.weight_s
            w_scale_to_use = self.w_scale_s
            ks_active = ks_cfg 
            groups_active = groups_cfg
            oup_for_w_scale_view = oup_actual
        else:
            assert x.shape[-1] == self.wh
            weight_to_use = self.get_active_filter(self.weight, inp, oup_actual, ks_cfg, groups_cfg).contiguous()
            w_scale_to_use = self.w_scale[:oup_actual]
            ks_active = ks_cfg
            groups_active = groups_cfg
            oup_for_w_scale_view = oup_actual
        
        x_processed = x
        if self.a_bit != 32 and self.a_bit is not None:
            if _ONNX_EXPORTING:
                a_s = self.a_scale.data 
                a_z = self.a_zero_point.data
                x_scaled = (x_processed - a_z) / a_s
                x_rounded = torch.round(x_scaled) 
                x_clamped = x_rounded.clamp(self.a_min_q, self.a_max_q)
                x_processed = x_clamped * a_s + a_z
            else: 
                if self.training and self.a_init_state == 0:
                    self.a_init_state.fill_(1); x_detach = x_processed.detach(); x_max_val = x_detach.max(); x_min_val = x_detach.min(); best_score = 1e+10; res_scale_val = self.a_scale.data.clone(); res_zero_point_val = self.a_zero_point.data.clone()
                    for i_act_init in range(80):
                        new_ub = x_max_val*(1.0-(i_act_init*0.01)); new_lb = x_min_val*(1.0-(i_act_init*0.01)); new_scale_val_init = (new_ub-new_lb)/(2**self.a_bit-1); new_zero_point_val_init = new_lb
                        x_q_val = (torch.round((x_detach-new_zero_point_val_init)/new_scale_val_init)).clamp(self.a_min_q,self.a_max_q)*new_scale_val_init + new_zero_point_val_init # Renamed x_q
                        score = lp_loss(x_detach,x_q_val,p=2.0,reduction='all')
                        if score < best_score: best_score=score; res_scale_val=new_scale_val_init; res_zero_point_val=new_zero_point_val_init
                    self.a_scale.data.copy_(res_scale_val); self.a_zero_point.data.copy_(res_zero_point_val)                
                self.a_scale.data.clamp_(self.eps); g_act = 1.0/math.sqrt(x_processed.numel()*self.a_max_q)
                current_a_scale_val = grad_scale(self.a_scale,g_act); current_a_zero_point_val = grad_scale(self.a_zero_point,g_act)
                arg_for_round_act = (x_processed-current_a_zero_point_val)/current_a_scale_val
                print(f"DEBUG ONNX (Activation Quant): Layer: {self.__class__.__name__}, Type: {type(arg_for_round_act)}, Shape: {arg_for_round_act.shape if hasattr(arg_for_round_act, 'shape') else 'N/A'}")
                x_processed = round_pass(arg_for_round_act).clamp(self.a_min_q,self.a_max_q)*current_a_scale_val + current_a_zero_point_val

        weight_final = weight_to_use
        if self.w_bit != 32:
            if _ONNX_EXPORTING: 
                w_s = w_scale_to_use.data
                if self.scale_channel != 1:
                    w_s = w_s.view(oup_for_w_scale_view, 1, 1, 1)
                weight_scaled = weight_to_use / w_s
                weight_rounded = torch.round(weight_scaled) 
                weight_clamped = weight_rounded.clamp(self.w_min_q, self.w_max_q)
                weight_final = weight_clamped * w_s
            else: 
                if self.training and self.w_init_state == 0: 
                    self.w_init_state.fill_(1); w_detach=weight_to_use.detach()
                    if self.scale_channel==1: w_mean=w_detach.mean(); w_std=w_detach.std()
                    else: dim_w=[i_w for i_w in range(1,w_detach.dim())]; w_mean=w_detach.mean(dim=dim_w); w_std=w_detach.std(dim=dim_w)
                    v1=torch.abs(w_mean-3*w_std); v2=torch.abs(w_mean+3*w_std); w_scale_to_use.data.copy_(torch.max(v1,v2)/2**(self.w_bit-1))
                w_scale_to_use.data.clamp_(self.eps); g_weight = 1.0/math.sqrt(weight_to_use.numel()*self.w_max_q)
                current_w_scale_val = grad_scale(w_scale_to_use,g_weight)
                if self.scale_channel != 1: current_w_scale_val = current_w_scale_val.view(oup_for_w_scale_view,1,1,1)
                arg_for_round_weight = weight_to_use / current_w_scale_val
                print(f"DEBUG ONNX (Weight Quant): Layer: {self.__class__.__name__}, Type: {type(arg_for_round_weight)}, Shape: {arg_for_round_weight.shape if hasattr(arg_for_round_weight, 'shape') else 'N/A'}")
                weight_final = round_pass(arg_for_round_weight).clamp(self.w_min_q,self.w_max_q) * current_w_scale_val
        
        padding = (ks_active - 1) // 2 * self.dilation
        out = F.conv2d(x_processed, weight_final, None, stride=self.stride, padding=padding,
                       dilation=self.dilation, groups=groups_active)
        return out

    def to_static(self, x, sub_path):
        self.static = True
        assert x.shape[-1] == self.wh
        oup, ks, groups = sub_path
        inp = x.shape[1]
        if oup == -1:
            oup = inp
        weight_param = self.get_active_filter(self.weight, inp, oup, ks, groups).contiguous() # Renamed weight
        self.weight.requires_grad = False
        if hasattr(self, 'matrix_7to5'):
            self.matrix_7to5.requires_grad = False
        if hasattr(self, 'matrix_5to3'):
            self.matrix_5to3.requires_grad = False
        self.register_parameter('weight_s', nn.Parameter(weight_param.data))
        if self.w_bit != 32:
            w_scale_val_static = None
            if self.scale_channel != 1:
                w_scale_val_static = self.w_scale[:oup].data
            else:
                w_scale_val_static = self.w_scale.data
            self.w_scale.requires_grad = False
            self.register_parameter('w_scale_s', nn.Parameter(w_scale_val_static))
        
        if hasattr(self, 'a_scale'): self.a_scale.requires_grad = False
        if hasattr(self, 'a_zero_point'): self.a_zero_point.requires_grad = False
        
        return self.forward(x, sub_path)

    def get_flops_bitops(self, sub_path):
        if self.max_inp == 3: 
            if tuple(sub_path) not in self.ops_memory:
                pre_channels = 3 
                channels, ks, groups = sub_path
                bitops = 0.0 
                a_bit_eff = self.a_bit if self.a_bit is not None else 8 
                w_bit_eff = self.w_bit if self.w_bit != 32 else 32 
                if self.a_bit is None and self.w_bit == 8: 
                    factor = (64 / math.sqrt(8 * 32)) 
                elif self.a_bit is not None : 
                    factor = (64 / math.sqrt(w_bit_eff * a_bit_eff))
                else: 
                    factor = 1 
                flops = (ks * ks * pre_channels / groups * channels * self.wh /
                         self.stride * self.wh / self.stride / factor)
                self.ops_memory[tuple(sub_path)] = (flops / 1e6, bitops / 1e6) 
            return self.ops_memory[tuple(sub_path)]
        else:
            return (0,0)

class DynamicFPLinear(nn.Module):
    def __init__(self, max_inp, oup):
        super().__init__()
        self.max_inp = max_inp
        self.oup = oup
        shape = (oup, max_inp)
        self.register_parameter('weight', nn.Parameter(torch.rand(shape)))
        self.register_parameter('bias', nn.Parameter(torch.rand(oup)))
        self.reset_parameters()
        self.ops_memory = {}
        self.static = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.static:
            filters = self.weight_s
        else:
            inp = x.shape[1]
            filters = self.weight[:, :inp].contiguous()
        out = F.linear(x, filters, self.bias)
        return out

    def to_static(self, x):
        self.static = True
        inp = x.shape[1]
        filters = self.weight[:, :inp].contiguous()
        self.weight.requires_grad = False
        self.register_parameter('weight_s', nn.Parameter(filters.data)) # Use .data
        return self.forward(x)

    def get_flops_bitops(self, pre_sub_path):
        if tuple(pre_sub_path) not in self.ops_memory:
            pre_channels = pre_sub_path[0]
            bitops = 0.0
            flops = self.oup * (pre_channels + 1)
            self.ops_memory[tuple(pre_sub_path)] = (flops / 1e6, bitops / 1e6)
        return self.ops_memory[tuple(pre_sub_path)]


class DynamicQLinear(nn.Module):
    def __init__(self, max_inp, max_oup, w_bit, a_bit, channel_wise=True):
        super().__init__()
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.w_bit = w_bit
        self.a_bit = a_bit
        shape = (max_oup, max_inp)
        self.register_parameter('weight', nn.Parameter(torch.rand(shape)))
        self.register_parameter('bias', nn.Parameter(torch.rand(max_oup)))

        if w_bit != 32:
            if channel_wise:
                self.scale_channel = max_oup
            else:
                self.scale_channel = 1
            self.register_parameter(
                'w_scale', nn.Parameter(torch.ones(self.scale_channel)))
            self.register_buffer('w_init_state', torch.zeros(1))
            self.w_min_q = -1 * 2**(w_bit - 1)
            self.w_max_q = 2**(w_bit - 1) - 1

        if a_bit != 32: # a_bit can be None
            self.register_parameter('a_scale', nn.Parameter(torch.ones(1)))
            self.register_parameter('a_zero_point',
                                    nn.Parameter(torch.ones(1)))
            self.register_buffer('a_init_state', torch.zeros(1))
            self.a_min_q = 0
            self.a_max_q = 2**a_bit - 1 if a_bit is not None else 0 # Handle a_bit is None

        self.eps = torch.finfo(torch.float32).eps
        self.reset_parameters()
        self.ops_memory = {}
        self.static = False
        # self.onnx_export_mode = False # We will use the global _utils._ONNX_EXPORTING

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, oup=None):
        weight_to_use = None
        bias_to_use = None
        w_scale_to_use = None
        oup_actual = oup

        if self.static:
            weight_to_use = self.weight_s
            bias_to_use = self.bias_s
            w_scale_to_use = self.w_scale_s
            oup_actual = self.weight_s.shape[0]
        else:
            inp = x.shape[1]
            oup_actual = oup if oup is not None else self.weight.shape[0]
            weight_to_use = self.weight[:oup_actual, :inp].contiguous()
            bias_to_use = self.bias[:oup_actual]
            w_scale_to_use = self.w_scale[:oup_actual]
        
        x_processed = x
        if self.a_bit != 32 and self.a_bit is not None: # Check a_bit is not None
            if _ONNX_EXPORTING: 
                a_s = self.a_scale.data 
                a_z = self.a_zero_point.data
                x_scaled = (x_processed - a_z) / a_s
                x_rounded = torch.round(x_scaled) 
                x_clamped = x_rounded.clamp(self.a_min_q, self.a_max_q)
                x_processed = x_clamped * a_s + a_z
            else: 
                if self.training and self.a_init_state == 0: 
                    self.a_init_state.fill_(1); x_detach=x_processed.detach(); x_max_val=x_detach.max(); x_min_val=x_detach.min(); best_score=1e+10; res_scale_val=self.a_scale.data.clone(); res_zero_point_val=self.a_zero_point.data.clone()
                    for i_act_init in range(80):
                        new_ub=x_max_val*(1.0-(i_act_init*0.01)); new_lb=x_min_val*(1.0-(i_act_init*0.01)); new_scale_val_init=(new_ub-new_lb)/(2**self.a_bit-1); new_zero_point_val_init=new_lb
                        x_q_val=(torch.round((x_detach-new_zero_point_val_init)/new_scale_val_init)).clamp(self.a_min_q,self.a_max_q)*new_scale_val_init+new_zero_point_val_init
                        score=lp_loss(x_detach,x_q_val,p=2.0,reduction='all')
                        if score<best_score: best_score=score; res_scale_val=new_scale_val_init; res_zero_point_val=new_zero_point_val_init
                    self.a_scale.data.copy_(res_scale_val); self.a_zero_point.data.copy_(res_zero_point_val)
                self.a_scale.data.clamp_(self.eps); g_act=1.0/math.sqrt(x_processed.numel()*self.a_max_q)
                current_a_scale_val=grad_scale(self.a_scale,g_act); current_a_zero_point_val=grad_scale(self.a_zero_point,g_act)
                arg_for_round_act = (x_processed-current_a_zero_point_val)/current_a_scale_val
                # print(f"DEBUG ONNX (Activation Quant): Layer: {self.__class__.__name__}, Type: {type(arg_for_round_act)}, Shape: {arg_for_round_act.shape if hasattr(arg_for_round_act, 'shape') else 'N/A'}")
                x_processed = round_pass(arg_for_round_act).clamp(self.a_min_q,self.a_max_q)*current_a_scale_val + current_a_zero_point_val

        weight_final = weight_to_use
        if self.w_bit != 32:
            if _ONNX_EXPORTING: 
                w_s = w_scale_to_use.data
                if self.scale_channel != 1:
                    w_s = w_s.view(oup_actual, 1)
                weight_scaled = weight_to_use / w_s
                print(f"DEBUG ONNX (Weight Quant - ONNX PATH): Layer: {self.__class__.__name__}, Type(weight_scaled): {type(weight_scaled)}, Shape: {weight_scaled.shape if hasattr(weight_scaled, 'shape') else 'N/A'}")
                weight_rounded = torch.round(weight_scaled)
                weight_clamped = weight_rounded.clamp(self.w_min_q, self.w_max_q)
                weight_final = weight_clamped * w_s
            else: 
                if self.training and self.w_init_state == 0: 
                    self.w_init_state.fill_(1); w_detach=weight_to_use.detach()
                    if self.scale_channel==1: w_mean=w_detach.mean(); w_std=w_detach.std()
                    else: dim_w=[i_w for i_w in range(1,w_detach.dim())]; w_mean=w_detach.mean(dim=dim_w); w_std=w_detach.std(dim=dim_w)
                    v1=torch.abs(w_mean-3*w_std); v2=torch.abs(w_mean+3*w_std); w_scale_to_use.data.copy_(torch.max(v1,v2)/2**(self.w_bit-1))
                w_scale_to_use.data.clamp_(self.eps); g_weight=1.0/math.sqrt(weight_to_use.numel()*self.w_max_q)
                current_w_scale_val=grad_scale(w_scale_to_use,g_weight)
                if self.scale_channel != 1: current_w_scale_val=current_w_scale_val.view(oup_actual,1)
                arg_for_round_weight = weight_to_use / current_w_scale_val
                # print(f"DEBUG ONNX (Weight Quant - NORMAL PATH): Layer: {self.__class__.__name__}, Type: {type(arg_for_round_weight)}, Shape: {arg_for_round_weight.shape if hasattr(arg_for_round_weight, 'shape') else 'N/A'}")
                weight_final = round_pass(arg_for_round_weight).clamp(self.w_min_q,self.w_max_q) * current_w_scale_val
        
        out = F.linear(x_processed, weight_final, bias_to_use)
        return out

    def to_static(self, x, oup=None):
        self.static = True
        inp = x.shape[1]
        oup_val = oup if oup is not None else self.weight.shape[0]
            
        filters = self.weight[:oup_val, :inp].contiguous()
        bias_s_val = self.bias[:oup_val] 
        
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.register_parameter('weight_s', nn.Parameter(filters.data)) 
        self.register_parameter('bias_s', nn.Parameter(bias_s_val.data)) 
        
        if self.w_bit != 32:
            w_scale_val_static = None 
            if self.scale_channel != 1:
                w_scale_val_static = self.w_scale[:oup_val].data 
            else:
                w_scale_val_static = self.w_scale.data 
            self.w_scale.requires_grad = False
            self.register_parameter('w_scale_s', nn.Parameter(w_scale_val_static))
        
        if hasattr(self, 'a_scale'): self.a_scale.requires_grad = False
        if hasattr(self, 'a_zero_point'): self.a_zero_point.requires_grad = False

        return self.forward(x, oup_val) 

    def get_flops_bitops(self, pre_sub_path):
        if tuple(pre_sub_path) not in self.ops_memory:
            pre_channels = pre_sub_path[0]
            a_bit_eff = self.a_bit if self.a_bit is not None else 32 
            w_bit_eff = self.w_bit if self.w_bit != 32 else 32 
            if a_bit_eff == 32 or w_bit_eff == 32 :
                 factor = 1.0
            else: 
                 factor = (64 / math.sqrt(w_bit_eff * a_bit_eff))
            flops = self.max_oup * (pre_channels + 1) / factor
            bitops = 0.0 
            self.ops_memory[tuple(pre_sub_path)] = (flops / 1e6, bitops / 1e6)
        return self.ops_memory[tuple(pre_sub_path)]


class DynamicBinConv2d(nn.Module):
    def __init__(self,
                 max_inp_channels_supernet, # Renamed for clarity
                 max_oup_channels_supernet, # Renamed for clarity
                 ks_list,
                 groups_list,
                 wh,
                 stride=1,
                 dilation=1):
        super().__init__()
        self.max_inp_dynamic = max_inp_channels_supernet # Max for supernet's self.weight
        self.max_oup_dynamic = max_oup_channels_supernet # Max for supernet's self.weight
        self.ks_list = list(set(ks_list))
        self.ks_list.sort()
        self.max_ks = max(ks_list)
        self.groups_list = groups_list # Potential choices for groups
        self.wh = wh # Original WH for this layer's position
        self.stride = stride
        self.dilation = dilation
        
        # self.weight is the large supernet weight tensor
        shape = (self.max_oup_dynamic, self.max_inp_dynamic // min(self.groups_list if self.groups_list else [1]), self.max_ks, self.max_ks)
        self.register_parameter('weight', nn.Parameter(torch.rand(shape)))
        self.register_buffer('scaling_factor', torch.ones(self.max_oup_dynamic, 1, 1, 1))

        if len(self.ks_list) > 1:
            # ... (matrix parameters for kernel transformation as before) ...
            scale_params = {}
            for i in range(len(self.ks_list) - 1):
                ks_small = self.ks_list[i]
                ks_larger = self.ks_list[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                scale_params['matrix_%s' % param_name] = nn.Parameter(torch.eye(ks_small**2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        if max(self.ks_list) > 1: # max_ks can be 1
            name = 'fp2bin_%d' % max(self.ks_list)
            if max(self.ks_list)**2 > 0:
                 self.register_parameter(name, nn.Parameter(torch.eye(max(self.ks_list)**2)))
            
        self.eps = torch.finfo(torch.float32).eps
        self.reset_parameters()
        self.is_bin = True
        self.is_distill = False
        self.distill_loss_func = nn.MSELoss()
        self.static_mode_active = False
        # self.weight_s and self.scaling_factor_s will be created in to_static

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def get_active_filter(self, base_weight_tensor, actual_inp_channels, actual_out_channels, target_ks, actual_groups):
        # base_weight_tensor is self.weight (supernet's large tensor)
        # actual_inp_channels, actual_out_channels, target_ks, actual_groups are for the specific static instance.
        
        # Determine kernel slice for target_ks
        start_k, end_k = _sub_filter_start_end(self.max_ks, target_ks)
        
        # Initial slice from supernet weight for max_inp_dynamic, max_oup_dynamic, and max_ks
        # Slice output channels first
        filters = base_weight_tensor[:actual_out_channels, :, :, :] # O, C_in_max/G_min, K_max, K_max
        
        # Slice input channels: C_in_max/G_min -> actual_inp_channels/actual_groups
        # This requires careful handling of groups in the supernet weight definition.
        # Supernet weight: (max_oup, max_inp // min_groups, max_ks, max_ks)
        # Static weight needed: (actual_out, actual_inp // actual_groups, target_ks, target_ks)
        
        # Effective input channels in the base_weight_tensor per output channel group
        # This assumes min_groups was used to define self.weight's second dim.
        # This part is complex if min_groups != 1. Let's assume min_groups=1 for simplicity in supernet weight channel def.
        # If supernet weight is (max_out, max_in, max_k, max_k) before any grouping considerations:
        #   filters_sliced_inp = filters[:, :actual_inp_channels, start_k:end_k, start_k:end_k]
        # If supernet weight is (max_out, max_in // G_min, max_k, max_k):
        #   The slice needs to be: actual_inp_channels // actual_groups
        
        # Let's assume self.weight is (max_oup, max_inp_per_group_in_supernet, max_ks, max_ks)
        # And we need to get (actual_out, actual_inp // actual_groups, target_ks, target_ks)
        
        inp_channels_per_group_static = actual_inp_channels // actual_groups
        
        # Slice from the supernet weight's input channel dimension
        filters = filters[:, :inp_channels_per_group_static, start_k:end_k, start_k:end_k]
        
        # Kernel transformation (if ks < self.max_ks)
        if target_ks < self.max_ks and len(self.ks_list) > 1:
            # This part of your original get_active_filter transforms kernels.
            # Ensure `filters` at this point is correctly shaped for transformation.
            # The transformation should work on (O, C_in_group, K_current, K_current) -> (O, C_in_group, K_target, K_target)
            # (The original kernel transformation logic from your provided code)
            # This part seems to iterate downwards from max_ks.
            # Let current_transformed_filter be the `filters` after channel and initial kernel slicing.
            current_transformed_filter = filters 
            for i_ks_transform in range(len(self.ks_list) - 1, 0, -1):
                src_ks_transform = self.ks_list[i_ks_transform]
                if src_ks_transform <= target_ks: # If current source is already target_ks or smaller, stop.
                    break
                target_ks_transform = self.ks_list[i_ks_transform - 1]
                if target_ks_transform < target_ks: # Don't transform smaller than final target_ks
                    continue

                start_idx_k_transform, end_idx_k_transform = _sub_filter_start_end(src_ks_transform, target_ks_transform)
                _input_filter_for_matrix = current_transformed_filter[:, :, start_idx_k_transform:end_idx_k_transform, start_idx_k_transform:end_idx_k_transform]
                _input_filter_for_matrix = _input_filter_for_matrix.contiguous()
                _input_filter_for_matrix = _input_filter_for_matrix.view(
                    _input_filter_for_matrix.size(0), _input_filter_for_matrix.size(1), -1) # O, C_group, src_ks_t^2
                _input_filter_for_matrix = _input_filter_for_matrix.view(-1, _input_filter_for_matrix.size(2)) # O*C_group, src_ks_t^2
                
                _transformed_filter_flat = F.linear(
                    _input_filter_for_matrix,
                    getattr(self, 'matrix_%dto%d' % (src_ks_transform, target_ks_transform))) # O*C_group, target_ks_t^2
                
                current_transformed_filter = _transformed_filter_flat.view(
                    actual_out_channels, inp_channels_per_group_static, target_ks_transform, target_ks_transform)
            filters = current_transformed_filter # This is now (O, C_in_group, target_ks, target_ks)

        # The grouping by chunking output channels and then slicing input channels from original code was complex.
        # If `F.conv2d` is used with `groups=actual_groups`, the weight tensor `filters`
        # should directly be of shape (actual_out_channels, actual_inp_channels // actual_groups, target_ks, target_ks).
        # The slicing above aims to achieve this.
        return filters.contiguous()


    def fp2bin_filter(self, weight_to_bin): # weight is (O, C_in_group, K, K)
        oc, ic_g, ks_val, _ = weight_to_bin.shape
        if ks_val > 1 and hasattr(self, f'fp2bin_{ks_val}'): 
            weight_view1 = weight_to_bin.view(oc, ic_g, -1) 
            weight_view2 = weight_view1.view(-1, ks_val**2) 
            weight_linear = F.linear(weight_view2, getattr(self, f'fp2bin_{ks_val}')) 
            weight_view3 = weight_linear.view(oc, ic_g, ks_val**2) 
            weight_final_out = weight_view3.view(oc, ic_g, ks_val, ks_val) 
            return weight_final_out
        return weight_to_bin

    def forward(self, x, loss=None, sub_path=None, **kwargs):
        # ... (sub_path and loss initialization as in previous correct version) ...
        if sub_path is None and 'sub_path' in kwargs: sub_path = kwargs['sub_path']
        if sub_path is None: raise ValueError("DynamicBinConv2d.forward expects 'sub_path'")

        current_iter_loss = None
        if not _utils._ONNX_EXPORTING:
            # Initialize loss logic from previous version
            if loss is None and 'loss' in kwargs: current_iter_loss = kwargs['loss']
            elif loss is not None: current_iter_loss = loss
            if current_iter_loss is None: current_iter_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # sub_path for DynamicBinConv2d: [actual_out_channels, target_ks, actual_groups]
        # Note: actual_inp_channels comes from x.shape[1]
        actual_out_channels_cfg, target_ks_cfg, actual_groups_cfg = sub_path
        actual_inp_channels_runtime = x.shape[1]
        
        # If oup_cfg is -1 (e.g. from BasicBlock's first binary_conv), it means oup = inp.
        # This actual_out_channels needs to be used for sizing weights if dynamic.
        # For static, weight_s is already sized.
        resolved_actual_out_channels = actual_inp_channels_runtime if actual_out_channels_cfg == -1 else actual_out_channels_cfg

        active_weight_for_conv_runtime = None
        current_scaling_factor_runtime = None

        if self.static_mode_active and hasattr(self, 'weight_s') and hasattr(self, 'scaling_factor_s'):
            active_weight_for_conv_runtime = self.weight_s
            current_scaling_factor_runtime = self.scaling_factor_s
        else: # Dynamic supernet mode
            # For dynamic mode, get_active_filter needs the runtime input channels and configured output/ks/groups
            active_weight_from_super = self.get_active_filter(
                self.weight, actual_inp_channels_runtime, resolved_actual_out_channels, target_ks_cfg, actual_groups_cfg)
            if self.is_bin:
                active_weight_from_super = self.fp2bin_filter(active_weight_from_super)
            active_weight_for_conv_runtime = active_weight_from_super.contiguous()
            
            # Dynamic scaling factor
            sf_slice_dynamic = self.scaling_factor[:resolved_actual_out_channels]
            # (Scaling factor update logic for training as before)
            # For forward pass, current_scaling_factor_runtime uses this slice.
            current_scaling_factor_runtime = sf_slice_dynamic


        if _utils._ONNX_EXPORTING:
            # ONNX Path (uses self.weight_s, self.scaling_factor_s)
            w_mean_onnx = active_weight_for_conv_runtime.mean(dim=(1, 2, 3), keepdim=True)
            w_std_onnx = active_weight_for_conv_runtime.std(dim=(1, 2, 3), keepdim=True) + self.eps
            weight_n_onnx = (active_weight_for_conv_runtime - w_mean_onnx) / w_std_onnx
            binary_weight_approx = current_scaling_factor_runtime.detach() * torch.sign(weight_n_onnx)
            
            padding = (target_ks_cfg - 1) // 2 * self.dilation
            out = F.conv2d(x, binary_weight_approx, None,
                           stride=self.stride, padding=padding, dilation=self.dilation, groups=actual_groups_cfg)
            return out 
        else:
            # Training/Normal Eval Path
            w_mean = active_weight_for_conv_runtime.mean(dim=(1, 2, 3), keepdim=True)
            w_std = active_weight_for_conv_runtime.std(dim=(1, 2, 3), keepdim=True) + self.eps
            weight_n = (active_weight_for_conv_runtime - w_mean) / w_std
            
            binary_weight_final = None
            if self.is_bin:
                _sf_to_use_for_bin = None
                if not self.static_mode_active : # Dynamic training
                    # Update the slice of the main scaling_factor buffer
                    sf_slice_dynamic.copy_(torch.mean(abs(weight_n.data), dim=(1, 2, 3), keepdim=True))
                    _sf_to_use_for_bin = sf_slice_dynamic.detach()
                else: # Static mode (but not ONNX export, e.g. PyTorch eval of static model)
                    _sf_to_use_for_bin = current_scaling_factor_runtime.detach() # which is self.scaling_factor_s
                binary_weight_final = _sf_to_use_for_bin * SignFuncW.apply(weight_n)
            else:
                binary_weight_final = weight_n

            padding = (target_ks_cfg - 1) // 2 * self.dilation
            out = F.conv2d(x, binary_weight_final, None, 
                           stride=self.stride, padding=padding, dilation=self.dilation, groups=actual_groups_cfg)
            
            if self.is_distill:
                # ... (distillation loss as before) ...
                distill_weight = weight_n 
                binary_weight_n_for_distill = binary_weight_final / (binary_weight_final.abs().pow(2).mean(dim=(1,2,3),keepdim=True) + self.eps)
                with torch.no_grad():
                    distill_weight_n = distill_weight / (distill_weight.abs().pow(2).mean(dim=(1,2,3),keepdim=True) + self.eps)
                current_iter_loss += 0.1 * self.distill_loss_func(binary_weight_n_for_distill, distill_weight_n.detach())

            return out, current_iter_loss

    def to_static(self, x_dummy_for_shape, loss_dummy, sub_path_static):
        # sub_path_static: [actual_out_channels, target_ks, actual_groups] for this layer
        # x_dummy_for_shape: provides the actual_inp_channels for this static layer instance.
        self.static_mode_active = True
        
        actual_out_channels_s, target_ks_s, actual_groups_s = sub_path_static
        actual_inp_channels_s = x_dummy_for_shape.shape[1] # Input channels for THIS static layer

        # If actual_out_channels_s is -1, it means output channels = input channels for this layer
        if actual_out_channels_s == -1:
            actual_out_channels_s = actual_inp_channels_s
            # Update sub_path_static for the forward call if it's used there with this resolved value
            # However, forward typically gets sub_path directly. This is for sizing weight_s.

        # Get the specific weight configuration for the static model using actual dimensions
        weight_data_for_static_s = self.get_active_filter(
            self.weight, actual_inp_channels_s, actual_out_channels_s, target_ks_s, actual_groups_s)
        
        if self.is_bin: 
            weight_data_for_static_s = self.fp2bin_filter(weight_data_for_static_s)
        
        # Register static weight parameter
        if hasattr(self, 'weight_s') and self.weight_s.shape == weight_data_for_static_s.shape:
            self.weight_s.data.copy_(weight_data_for_static_s.data)
        else:
            if hasattr(self, 'weight_s'): del self._parameters['weight_s'] # remove if shape mismatch
            self.register_parameter('weight_s', nn.Parameter(weight_data_for_static_s.data.clone()))
        
        # Create and register static scaling_factor_s
        # It should be derived based on the final static weights (weight_s)
        w_s_mean_static = self.weight_s.data.mean(dim=(1, 2, 3), keepdim=True)
        w_s_std_static = self.weight_s.data.std(dim=(1, 2, 3), keepdim=True) + self.eps
        w_s_n_static = (self.weight_s.data - w_s_mean_static) / w_s_std_static
        sf_s_data = torch.mean(abs(w_s_n_static.data), dim=(1, 2, 3), keepdim=True).to(x_dummy_for_shape.dtype)
        # sf_s_data should have shape (actual_out_channels_s, 1, 1, 1)

        if hasattr(self, 'scaling_factor_s') and self.scaling_factor_s.shape == sf_s_data.shape:
            self.scaling_factor_s.data.copy_(sf_s_data)
        else:
            if hasattr(self, 'scaling_factor_s'): del self._buffers['scaling_factor_s']
            self.register_buffer('scaling_factor_s', sf_s_data.clone())

        # Freeze original supernet parameters that are part of this layer
        self.weight.requires_grad = False
        if hasattr(self, 'matrix_7to5'): self.matrix_7to5.requires_grad = False
        # ... (freeze other transformation matrices and fp2bin matrices as before) ...
        if max(self.ks_list) > 1 and hasattr(self, f'fp2bin_{max(self.ks_list)}'): # max_ks can be 1
            fp2bin_param = getattr(self, f'fp2bin_{max(self.ks_list)}')
            if isinstance(fp2bin_param, nn.Parameter): fp2bin_param.requires_grad = False
        
        # The forward call here is to trace/test the static path with correctly sized static params
        # It uses the sub_path_static passed, which defines the config for weight_s
        if _utils._ONNX_EXPORTING:
            # Forward will use self.weight_s and self.scaling_factor_s due to self.static_mode_active
            return self.forward(x_dummy_for_shape, sub_path=sub_path_static) 
        else:
            return self.forward(x_dummy_for_shape, loss=loss_dummy, sub_path=sub_path_static)
        

class DynamicPReLU(nn.Module):
    def __init__(self, max_channels):
        super().__init__()
        self.max_channels = max_channels
        self.prelu = nn.PReLU(max_channels)
        self.static = False

    def forward(self, x):
        feature_dim = x.size(1)
        if self.static:
            out = self.prelu_s(x)
        else:
            out = F.prelu(x, self.prelu.weight[:feature_dim])
        return out

    def to_static(self, x):
        self.static = True
        feature_dim = x.size(1)
        weight_val = self.prelu.weight[:feature_dim] 
        self.prelu.weight.requires_grad = False
        self.prelu_s = nn.PReLU(feature_dim)
        self.prelu_s.eval()
        self.prelu_s.weight.data.copy_(weight_val.data) # Use .data
        return self.forward(x)


class DynamicBatchNorm2d(nn.Module):
    def __init__(self, max_channels):
        super().__init__()
        self.max_channels = max_channels
        self.bn = nn.BatchNorm2d(max_channels)
        self.static = False

    def forward(self, x):
        feature_dim = x.size(1)
        bn_to_use = None 
        if self.static:
            bn_to_use = self.bn_s
        else:
            bn_to_use = self.bn
        
        exponential_average_factor = 0.0
        if bn_to_use.momentum is not None:
            exponential_average_factor = bn_to_use.momentum

        if bn_to_use.training and bn_to_use.track_running_stats:
            if bn_to_use.num_batches_tracked is not None:
                bn_to_use.num_batches_tracked.add_(1)
                if bn_to_use.momentum is None: 
                    exponential_average_factor = 1.0 / float(
                        bn_to_use.num_batches_tracked)
                else:  
                    exponential_average_factor = bn_to_use.momentum

        bn_training_mode = bn_to_use.training 
        if not bn_to_use.training: 
            bn_training_mode = (bn_to_use.running_mean is None) and (bn_to_use.running_var is None)
        
        running_mean_to_use = bn_to_use.running_mean
        running_var_to_use = bn_to_use.running_var
        weight_to_use_bn = bn_to_use.weight 
        bias_to_use_bn = bn_to_use.bias 

        if not self.static: 
            if running_mean_to_use is not None: running_mean_to_use = running_mean_to_use[:feature_dim]
            if running_var_to_use is not None: running_var_to_use = running_var_to_use[:feature_dim]
            if weight_to_use_bn is not None: weight_to_use_bn = weight_to_use_bn[:feature_dim]
            if bias_to_use_bn is not None: bias_to_use_bn = bias_to_use_bn[:feature_dim]

        return F.batch_norm(
            x,
            running_mean_to_use if not bn_to_use.training or bn_to_use.track_running_stats else None,
            running_var_to_use if not bn_to_use.training or bn_to_use.track_running_stats else None,
            weight_to_use_bn,
            bias_to_use_bn,
            bn_training_mode, 
            exponential_average_factor,
            bn_to_use.eps,
        )

    def to_static(self, x):
        self.static = True
        feature_dim = x.size(1)
        running_mean = self.bn.running_mean[:feature_dim].data 
        running_var = self.bn.running_var[:feature_dim].data   
        weight_val = self.bn.weight[:feature_dim].data        
        bias_val = self.bn.bias[:feature_dim].data            
        
        self.bn.weight.requires_grad = False
        self.bn.bias.requires_grad = False
        
        self.bn_s = nn.BatchNorm2d(feature_dim)
        self.bn_s.eval() 
        self.bn_s.running_mean.copy_(running_mean)
        self.bn_s.running_var.copy_(running_var)
        self.bn_s.weight.data.copy_(weight_val)
        self.bn_s.bias.data.copy_(bias_val)
        return self.forward(x)
    
class DynamicPReLU(nn.Module):
    def __init__(self, max_channels):
        super().__init__()
        self.max_channels = max_channels
        self.prelu = nn.PReLU(max_channels)
        self.static = False

    def forward(self, x):
        feature_dim = x.size(1)
        if self.static:
            out = self.prelu_s(x)
        else:
            out = F.prelu(x, self.prelu.weight[:feature_dim])
        return out

    def to_static(self, x):
        self.static = True
        feature_dim = x.size(1)
        weight_val = self.prelu.weight[:feature_dim] # Renamed weight
        self.prelu.weight.requires_grad = False
        self.prelu_s = nn.PReLU(feature_dim)
        self.prelu_s.eval()
        self.prelu_s.weight.data.copy_(weight_val.data)
        return self.forward(x)


class DynamicBatchNorm2d(nn.Module):
    def __init__(self, max_channels):
        super().__init__()
        self.max_channels = max_channels
        self.bn = nn.BatchNorm2d(max_channels)
        self.static = False

    def forward(self, x):
        feature_dim = x.size(1)
        bn_to_use = None # Renamed bn
        if self.static:
            bn_to_use = self.bn_s
        else:
            bn_to_use = self.bn
        
        exponential_average_factor = 0.0
        if bn_to_use.momentum is not None:
            exponential_average_factor = bn_to_use.momentum

        if bn_to_use.training and bn_to_use.track_running_stats:
            if bn_to_use.num_batches_tracked is not None:
                bn_to_use.num_batches_tracked.add_(1)
                if bn_to_use.momentum is None: 
                    exponential_average_factor = 1.0 / float(
                        bn_to_use.num_batches_tracked)
                else:  
                    exponential_average_factor = bn_to_use.momentum

        bn_training_mode = bn_to_use.training # Renamed bn_training
        if not bn_to_use.training: # If not in training mode (eval)
             # Check if running_mean and running_var exist (they should for eval if track_running_stats is True)
            bn_training_mode = (bn_to_use.running_mean is None) and (bn_to_use.running_var is None)
        
        running_mean_to_use = bn_to_use.running_mean
        running_var_to_use = bn_to_use.running_var
        weight_to_use_bn = bn_to_use.weight # Renamed weight
        bias_to_use_bn = bn_to_use.bias # Renamed bias

        if not self.static: # Slice parameters only if not static (static version already has correct size)
            if running_mean_to_use is not None: running_mean_to_use = running_mean_to_use[:feature_dim]
            if running_var_to_use is not None: running_var_to_use = running_var_to_use[:feature_dim]
            if weight_to_use_bn is not None: weight_to_use_bn = weight_to_use_bn[:feature_dim]
            if bias_to_use_bn is not None: bias_to_use_bn = bias_to_use_bn[:feature_dim]

        return F.batch_norm(
            x,
            running_mean_to_use if not bn_to_use.training or bn_to_use.track_running_stats else None,
            running_var_to_use if not bn_to_use.training or bn_to_use.track_running_stats else None,
            weight_to_use_bn,
            bias_to_use_bn,
            bn_training_mode, # Use the derived training_mode flag
            exponential_average_factor,
            bn_to_use.eps,
        )

    def to_static(self, x):
        self.static = True
        feature_dim = x.size(1)
        running_mean = self.bn.running_mean[:feature_dim].data # Use .data
        running_var = self.bn.running_var[:feature_dim].data   # Use .data
        weight_val = self.bn.weight[:feature_dim].data        # Use .data
        bias_val = self.bn.bias[:feature_dim].data            # Use .data
        
        self.bn.weight.requires_grad = False
        self.bn.bias.requires_grad = False
        
        self.bn_s = nn.BatchNorm2d(feature_dim)
        self.bn_s.eval() # Important to set static BN to eval mode
        self.bn_s.running_mean.copy_(running_mean)
        self.bn_s.running_var.copy_(running_var)
        self.bn_s.weight.data.copy_(weight_val)
        self.bn_s.bias.data.copy_(bias_val)
        return self.forward(x)