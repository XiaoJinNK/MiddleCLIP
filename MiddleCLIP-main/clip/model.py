### CLIP source code from OpenAI:
# https://github.com/openai/CLIP/blob/main/clip/clip.py

from collections import OrderedDict
from typing import Tuple, Union

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import time
import os

from pamr import PAMR
import logging
import cv2


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.width = width
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.arch, self.attn_strategy, self.gaussian_std = None, None, 0
        self.addition_cache = dict()

        self.attn_save_dir = None

        self.mid_layer = [11] 
        self.use_shallow_fusion = False  
        self.shallow_layer_idx = None  
        self.shallow_fusion_weight = None  
        self.collected_shallow_feature = None  
        self.use_branch = False  
        self.branch_layers_from_end = None  
        self.branch_fusion_weight = None  

    # MiddleCLIP_arch:
    # 'wo_resi': without residual (in NACLIP, ClearCLIP)
    # 'w_resi': with residual (in CLIP, SCLIP)
    # attn_strategy:
    # 'vanilla': CLIP, 'sclip': SCLIP, 'naclip': NACLIP, 'clearclip': ClearCLIP, 'middleclip': MiddleCLIP
    # if use_rcs, use_sfr = True, True --> ResCLIP
    # if use_rcs, use_sfr, use_shallow_fusion, use_branch = True, True, True, True --> MiddleCLIP
    def set_params(self, arch, attn_strategy, gaussian_std):
        """Set model architecture and attention strategy
    
        Args:
            arch: Either 'w_resi' or 'wo_resi' for residual connections
            attn_strategy: One of ['vanilla', 'sclip', 'naclip', 'clearclip', 'middleclip']
            gaussian_std: Standard deviation for Gaussian smoothing
        """
        assert arch in ['w_resi', 'wo_resi']
        assert attn_strategy != 'sclip' or arch == 'w_resi'  # SCLIP setting
        assert gaussian_std > 0
        self.arch, self.attn_strategy, self.gaussian_std = arch, attn_strategy, gaussian_std



    def create_patch_matrix(self, seg_logit_temp, n_patches, temp_thd=None):
        seg_logit_temp = seg_logit_temp.cuda()
        seg_logit_temp = seg_logit_temp.squeeze(0)  # -->(150,224,224) & (30,336,459)
        seg_pred = seg_logit_temp.argmax(0, keepdim=True)  # (150,224,224)->(1,224,224)

        # AdaThr
        if temp_thd is not None:
            seg_pred[seg_logit_temp.max(0, keepdim=True)[0] < temp_thd] = 0
        else:
            raise NotImplemented

        seg_pred = seg_pred.squeeze(0)

        # extract patches  (21,28,16,16)
        patches = seg_pred.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
        # reshape patches to 2D tensor
        patches = patches.reshape(n_patches[0], n_patches[1], -1)
        # find most possible category of each patch
        patch_matrix = torch.mode(patches, dim=-1)[0]

        return patch_matrix



    @staticmethod
    @torch.jit.script
    def check_path(start_coords: torch.Tensor, end_coords: torch.Tensor,
                   patch_logit_temp: torch.Tensor) -> torch.Tensor:
        # V: the function of mask where V_mn = 1, if there is a valid path from (m,n) to (i', j')
        steps = torch.max(torch.abs(end_coords - start_coords), dim=1)[0]
        max_steps = steps.max().item()
        t = torch.linspace(0, 1, steps=max_steps + 1, device=start_coords.device).unsqueeze(1).unsqueeze(0)

        current_coords = start_coords.unsqueeze(1) * (1 - t) + end_coords.unsqueeze(1) * t
        current_coords = current_coords.round().long()

        valid_steps = torch.arange(max_steps + 1, device=start_coords.device).unsqueeze(0) <= steps.unsqueeze(1)

        start_values = patch_logit_temp[start_coords[:, 0], start_coords[:, 1]].unsqueeze(1)
        current_values = patch_logit_temp[current_coords[:, :, 0], current_coords[:, :, 1]]

        path_valid = (current_values == start_values) | ~valid_steps
        return path_valid.all(dim=1)



    def generate_attention_sfr(self, patch_logit_temp, n_patches, gaussian_std=1.0, adjust_for_cls=True, delete_same_entity=True):
        '''
        patch_logit_temp：(14, 14) tensor，most possible category of each patch
        attn_sfr：if the category of patch i is same as patch j, attn_sfr[i,j]=1
        generate attn_sfr，each element (i,j) refer to similarity of patch i and patch j
        '''
        h, w = n_patches
        total_patches = h * w

        patch_logit_temp = patch_logit_temp.cuda()

        # matrix (total_patches + 1) x (total_patches + 1)
        # (588,588) --> (589,589) [cls]
        attention_sfr = torch.zeros((total_patches + 1, total_patches + 1), device=patch_logit_temp.device)

        patch_indices = torch.arange(total_patches, device=patch_logit_temp.device)
        i, j = torch.meshgrid(patch_indices, patch_indices, indexing='ij')
        i_w, i_h = torch.div(i, w, rounding_mode='floor'), i % w
        j_w, j_h = torch.div(j, w, rounding_mode='floor'), j % w

        mask = (patch_logit_temp[i_w, i_h] == patch_logit_temp[j_w, j_h])  # i: waiting to be compared with all patches, j: all patched
        attention_sfr[1:, 1:] = mask.float()

        # diagonal element set to 1 (self-attention)
        attention_sfr.diagonal().fill_(1)
        attention_sfr[0, :] = 0
        attention_sfr[:, 0] = 0

        # gaussian smoothing (only for matrix without [cls])
        non_cls_attention = attention_sfr[1:, 1:]

        # Del: adding a filtering, select the adjacent patch with aim patch of same category still to 1, others decaying.
        if delete_same_entity:
            # coords of start and end
            start_coords = torch.stack((i_w, i_h), dim=-1).view(-1, 2)
            end_coords = torch.stack((j_w, j_h), dim=-1).view(-1, 2)

            # check if belongs to same object
            valid_paths = VisionTransformer.check_path(start_coords, end_coords, patch_logit_temp)
            valid_paths = valid_paths.view(total_patches, total_patches)

            # different strategies of decaying
            # # 1. set to 0
            # non_cls_attention = non_cls_attention * valid_paths

            # # 2. set to 0.5 of origin
            # non_cls_attention = torch.where(valid_paths, non_cls_attention, non_cls_attention * 0.5)

            # 3. depend on distance to decay: Del_DelDecay
            # compute distance as steps
            steps = torch.max(torch.abs(end_coords - start_coords), dim=1)[0]
            max_steps = steps.max().item()
            # exponential decaying
            steps = steps.view(total_patches, total_patches)
            decay_factor = torch.exp(-steps / max_steps)
            # apply decaying
            # valid_paths (True): stay origin
            # valid_paths (False): decay with distance
            non_cls_attention = torch.where(valid_paths,
                                            non_cls_attention,
                                            non_cls_attention * decay_factor)

        # non_cls_attention_normalized = non_cls_attention / (non_cls_attention.sum(dim=-1, keepdim=True) + 1e-8)

        # 1D gaussian kernel
        kernel_size = min(total_patches, int(6 * gaussian_std + 1))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # make sure kernel_size is odd

        x = torch.arange(kernel_size, device=attention_sfr.device) - kernel_size // 2
        gaussian_kernel_1d = torch.exp(-(x ** 2) / (2 * gaussian_std ** 2))
        kernel = gaussian_kernel_1d / gaussian_kernel_1d.sum()

        # apply gaussian smoothing for every row, for every row use 1D convolution
        padding = kernel_size // 2  # compute padding size
        non_cls_attention_blurred = F.conv1d(
            non_cls_attention.unsqueeze(1),
            kernel.view(1, 1, -1),
            padding=padding
        ).squeeze(1)

        non_cls_attention_blurred = F.relu(non_cls_attention_blurred)
        row_sums = non_cls_attention_blurred.sum(dim=1, keepdim=True)
        non_cls_attention_normalized = non_cls_attention_blurred / (row_sums + 1e-8)

        # add [cls] to fit size
        attention_sfr[1:, 1:] = non_cls_attention_normalized
        attention_sfr = attention_sfr.unsqueeze(0)

        return attention_sfr



    def create_logit_temp(self, x, n_patches=None, attn_rcs=None, attn_rcs_weights=None, use_custom_attn=False,
                          query_features=None, img=None, patch_size=None, logit_scale=None, dtype=None, return_all=False,
                          align_corners=None):
        # compute seg logits
        blk = self.transformer.resblocks[-1]
        if self.arch == 'w_resi':
            x = x + self.custom_attn(blk.attn, blk.ln_1(x), n_patches=n_patches, attn_rcs=attn_rcs,
                                     attn_rcs_weights=attn_rcs_weights, use_custom_attn=use_custom_attn)
            x = x + blk.mlp(blk.ln_2(x))  # vanilla clip
        elif self.arch == 'wo_resi':
            x = self.custom_attn(blk.attn, blk.ln_1(x), n_patches=n_patches, attn_rcs=attn_rcs,
                                 attn_rcs_weights=attn_rcs_weights, use_custom_attn=use_custom_attn)
        else:
            raise NotImplemented

        x = x.permute(1, 0, 2)

        if return_all:
            x = self.ln_post(x) @ self.proj
        else:
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj

        # middleclip_segmentor
        image_features = x[:, 1:]  # (1,196,512)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ query_features.T

        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear', align_corners=align_corners)
        seg_logits_temp = logits

        # VOC20：(1,30,336,459) --> (1,30,336,459)
        pamr_simi_reuse = PAMR(10, dilations=(8, 16)).to(torch.device('cuda'))
        try:  # ade20K: seg_logits:[1,150,512,683] --> [1,150,512,683]
            seg_logits_temp = pamr_simi_reuse(img, seg_logits_temp.to(img.dtype)).to(dtype)
        except RuntimeError as e:
            logging.warning("")
            pass
        batch_size = seg_logits_temp.shape[0]
        for i in range(batch_size):
            seg_logits_temp = torch.softmax(seg_logits_temp[i] * logit_scale, dim=0)  # 1,n_queries * w * h -> n_queries,w,h
            seg_logits_temp = seg_logits_temp.unsqueeze(0)

        return seg_logits_temp



    def generate_guide_attention(self, collected_features, n_patches):
        """Generate key-to-key guide attention from shallow layer features
    
        Computes self-attention on keys from shallow layers to guide
        the complementary branch processing in SACF².
    
        Args:
            collected_features: Dictionary of features indexed by layer
            n_patches: Tuple of (height, width) for patch dimensions
    
        Returns:
            Guide attention matrix with shape [B*H, L, L]
            where B=batch, H=heads, L=sequence length
        """
        feature_layer_idx = self.shallow_layer_idx 
        attn_layer_idx = self.shallow_layer_idx  
        target_blk = self.transformer.resblocks[attn_layer_idx]
        guide_attn = self.custom_attn(target_blk.attn, target_blk.ln_1(collected_features[feature_layer_idx-1]), 
                                        n_patches=n_patches, with_attn=True, generate_guide_attn=True)[1]
        return guide_attn



    def forward(self, x: torch.Tensor, return_all=False, use_rcs=True, use_sfr=True, temp_thd=None,
                attn_sfr_weights=None, attn_rcs_weights=None, delete_same_entity=True, use_custom_attn=True,
                query_features=None, img=None, patch_size=None, logit_scale=None, dtype=None, align_corners=None):
        B, nc, w, h = x.shape
        n_patches = (w // self.patch_size, h // self.patch_size)
        x = self.conv1(x) 
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1) 
    
        x = torch.cat([
            self.class_embedding.to(x.dtype) + 
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
            x
        ], dim=1)
        
        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        attn_map = []  
        collected_features = {}
        self.collected_shallow_feature = None  
        input_branches = []  
        mid_x = None  
        mid_layer = self.mid_layer  
        layer_index = 0  
        first_end_layer = - self.branch_layers_from_end  
        
        
        # Process initial layers
        main_x = x  
        
        for i, blk in enumerate(self.transformer.resblocks[:first_end_layer]):
            if layer_index in mid_layer:
                x_i, attn_i = self.custom_attn(blk.attn, blk.ln_1(x), n_patches=n_patches, with_attn=True)
                mid_x = x_i + x
                mid_x = mid_x + blk.mlp(blk.ln_2(mid_x))
                attn_map.append(attn_i)
            else:
                x_i, attn_i = self.custom_attn(blk.attn, blk.ln_1(x), n_patches=n_patches, with_attn=True)
                attn_map.append(attn_i)
    
            x = x + x_i
            x = x + blk.mlp(blk.ln_2(x))
            main_x = x

            collected_features[layer_index] = main_x.clone()
            
            # Collect shallow layer feature for Local Preserved Middle Feature Fusion (LPMF²)
            if self.use_shallow_fusion and i == self.shallow_layer_idx:
                self.collected_shallow_feature = main_x[1:].clone() # remove CLS token
                
            layer_index += 1
            
        second_start_layer = first_end_layer 
        
        for i, blk in enumerate(self.transformer.resblocks[second_start_layer:-1]):
            if layer_index in mid_layer and mid_x is None:
                x_i, attn_i = self.custom_attn(blk.attn, blk.ln_1(x), n_patches=n_patches, with_attn=True)
                mid_x = x_i + x
                mid_x = mid_x + blk.mlp(blk.ln_2(mid_x))
                attn_map.append(attn_i)
            else:
                x_i, attn_i = self.custom_attn(blk.attn, blk.ln_1(x), n_patches=n_patches, with_attn=True)
                attn_map.append(attn_i)
            x = x + x_i
            x = x + blk.mlp(blk.ln_2(x))
            attn_map.append(attn_i)
            
            collected_features[layer_index] = main_x.clone()
            
            if self.use_branch:
                in_x = main_x + blk.mlp(blk.ln_2(main_x))
                input_branches.append(in_x)
            
            main_x = x  
            
            
            layer_index += 1
        
        # RCS(Residual Cross-correlation Self-attention from ResCLIP)
        attention_rcs = None
        if use_rcs and len(attn_map) >= 9:
            selected_attns = torch.stack([attn_map[i] for i in range(5, 9)])
            attention_rcs = selected_attns.mean(dim=0)
        
        # SFR(Semantic Feedback Refinement from ResCLIP)
        attention_sfr = None
        if use_sfr:
            seg_logit_temp = self.create_logit_temp(
                x, n_patches=n_patches, attn_rcs=attention_rcs,
                attn_rcs_weights=attn_rcs_weights, use_custom_attn=use_custom_attn,
                query_features=query_features, img=img, patch_size=patch_size,
                logit_scale=logit_scale, dtype=dtype, return_all=return_all,
                align_corners=align_corners
            )
            patch_logit_temp = self.create_patch_matrix(seg_logit_temp, n_patches, temp_thd=temp_thd)
            attention_sfr = self.generate_attention_sfr(
                patch_logit_temp, n_patches, gaussian_std=self.gaussian_std,
                delete_same_entity=delete_same_entity
            )

        # Local Preserved Middle Feature Fusion (LPMF²)
        if mid_x is not None and self.use_shallow_fusion is True and self.use_branch is True: 
            mid_x = self._apply_intermediate_feature_fusion(mid_x)
        elif self.use_branch is False and self.use_shallow_fusion is True:
            mid_x = self._apply_intermediate_feature_fusion(x) + mid_x
        elif self.use_branch is True and self.use_shallow_fusion is False:
            mid_x = x
        else:
            mid_x = x
             
        
        # Structure-Aware Complementary Feature Fusion (SACF²)
        # key to key guide attention
        guide_attn = self.generate_guide_attention(collected_features, n_patches)

        last_blk = self.transformer.resblocks[-1]
        lambda_2 = self.branch_fusion_weight
        # Final attention block with Relation Attention
        if self.arch == 'w_resi' and self.use_branch is True:
            out_current = mid_x + self.custom_attn(
                last_blk.attn, last_blk.ln_1(mid_x), n_patches=n_patches,
                attn_rcs=attention_rcs, attn_sfr=attention_sfr,
                attn_sfr_weights=attn_sfr_weights, attn_rcs_weights=attn_rcs_weights,
                use_custom_attn=use_custom_attn
            )
            out_current = out_current + last_blk.mlp(last_blk.ln_2(out_current))
            
        elif self.arch == 'w_resi' and self.use_branch is False:
            out_current = self.custom_attn(
                last_blk.attn, last_blk.ln_1(mid_x), n_patches=n_patches,
                attn_rcs=attention_rcs, attn_sfr=attention_sfr,
                attn_sfr_weights=attn_sfr_weights, attn_rcs_weights=attn_rcs_weights,
                use_custom_attn=use_custom_attn
            )
            out_current = out_current + last_blk.mlp(last_blk.ln_2(out_current))
            
        elif self.arch == 'wo_resi' and self.use_branch is True:
            out_current = self.custom_attn(
                last_blk.attn, last_blk.ln_1(mid_x), n_patches=n_patches,
                attn_rcs=attention_rcs, attn_sfr=attention_sfr,
                attn_sfr_weights=attn_sfr_weights, attn_rcs_weights=attn_rcs_weights,
                use_custom_attn=use_custom_attn
            )
            
        elif self.arch == 'wo_resi' and self.use_branch is False:
            out_current = self.custom_attn(
                last_blk.attn, last_blk.ln_1(mid_x), n_patches=n_patches,
                attn_rcs=attention_rcs, attn_sfr=attention_sfr,
                attn_sfr_weights=attn_sfr_weights, attn_rcs_weights=attn_rcs_weights,
                use_custom_attn=use_custom_attn
            )
        else:
            # vanilla clip
            out_current = x + self.custom_attn(
                last_blk.attn, last_blk.ln_1(x), n_patches=n_patches,
                attn_rcs=attention_rcs, attn_sfr=attention_sfr,
                attn_sfr_weights=attn_sfr_weights, attn_rcs_weights=attn_rcs_weights,
                use_custom_attn=use_custom_attn
            )
            out_current = out_current + last_blk.mlp(last_blk.ln_2(out_current))
        
        # Structure-Aware Complementary Feature Fusion (SACF²) SA fusion
        if input_branches and self.use_branch:
            sum_out_prev = self._process_branches(input_branches, last_blk, n_patches, guide_attn)
            sum_out = (1 - lambda_2) * out_current + lambda_2 * sum_out_prev
            x = sum_out
            
        elif self.use_shallow_fusion is True and self.use_branch is False:
            x = out_current  
             
        else:
            x = out_current
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        if return_all:
            return self.ln_post(x) @ self.proj
        
        x = self.ln_post(x[:, 0, :]) 
        if self.proj is not None:
            x = x @ self.proj
        
        return x
    
    def _apply_intermediate_feature_fusion(self, mid_x):
        """Apply Local Preserved Middle Feature Fusion (LPMF²)
    
        Fuses shallow layer features with middle layer features to preserve
        local semantic information across different network depths.
    
        Args:
            mid_x: Middle layer features with shape [L, N, D]
               where L includes CLS token
    
        Returns:
            Fused features with same shape as input
        """
        if not self.use_shallow_fusion or self.collected_shallow_feature is None:
            return mid_x
        mid_patch_features = mid_x[1:] 

        lambda_1 = self.shallow_fusion_weight
        fused_patch_features = (1 - lambda_1) * mid_patch_features + \
                             lambda_1 * self.collected_shallow_feature

        return torch.cat([mid_x[:1], fused_patch_features], dim=0)
    

   
    def _process_branches(self, input_branches, last_blk, n_patches, guide_attn):
        """Process Structure-Aware Complementary Feature Fusion (SACF²)
    
        Processes complementary branches using key-to-key guide attention
        to capture structure-aware semantic information.
    
        Args:
            input_branches: List of branch features from SAC blocks
            last_blk: Last transformer block for attention computation
            n_patches: Tuple of (height, width) for patch dimensions
            guide_attn: key to key guide attention from shallow layers
    
        Returns:
            Aggregated branch output with shape [L, N, D]
        """
        sum_out_prev = torch.zeros_like(input_branches[0])
        for in_x in input_branches:
            out_prev = self.custom_attn(
                last_blk.attn, last_blk.ln_1(in_x), n_patches=n_patches,
                use_custom_attn=False, guide_attn=guide_attn
            )

            sum_out_prev += out_prev
                
        return sum_out_prev



    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), mode='bicubic',
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)



    @staticmethod
    def gaussian_window(dim1, dim2, std=1.):  # (dim1,dim2)=(14,14)
        constant = 1 / (std * math.sqrt(2))  # 1/sqrt(2)
        ks = list()
        for dim in [dim1, dim2]:
            start = -(dim - 1) / 2.0
            k = torch.linspace(start=start * constant,
                               end=(start + (dim - 1)) * constant,
                               steps=dim,
                               dtype=torch.float)
            ks.append(k)
        dist_square_to_mu = (torch.stack(torch.meshgrid(*ks, indexing='ij')) ** 2).sum(0)
        return torch.exp(-dist_square_to_mu)



    @staticmethod
    def get_attention_addition(dim1, dim2, window=None, adjust_for_cls=True):
        # gaussian_window kernel, dim1=dim2=14
        m = torch.einsum('ij,kl->ijkl', torch.eye(dim1), torch.eye(dim2))  # (14,14,14,14)
        m = m.permute((0, 3, 1, 2)).contiguous()  # m[ijkl] = 1 iff (i, j) == (k, l)
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1), window.unsqueeze(0).unsqueeze(1), padding='same').squeeze(1)
        out = out.view(dim1 * dim2, dim1 * dim2)  # (196,196)
        if adjust_for_cls:  # (197,197) + [cls]
            v_adjusted = torch.vstack([torch.zeros((1, dim1 * dim2)), out])
            out = torch.hstack([torch.zeros((dim1 * dim2 + 1, 1)), v_adjusted])
        return out



    @staticmethod
    def calculate_distance_prior(x_origin):
        # cv2.distanceTransform for edge computing, doesn't work
        if isinstance(x_origin, torch.Tensor):
            x_origin = x_origin.cpu().numpy()
        if x_origin.dtype != np.uint8:
            x_origin = (x_origin * 255).astype(np.uint8)

        gray_image = cv2.cvtColor(x_origin, cv2.COLOR_RGB2GRAY)
        gray_image = gray_image.astype(np.uint8)
        _, binary_mask = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        distance_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        distance_transform_normalized = cv2.normalize(distance_transform, None, 0, 1.0, cv2.NORM_MINMAX)

        distance_transform_resized = cv2.resize(distance_transform_normalized, (197, 197))

        distance_tensor = torch.from_numpy(distance_transform_resized).float()
        return distance_tensor



    def custom_attn(self, attn_layer, x, return_attn=False, with_attn=False, n_patches=None, attn_rcs=None, attn_sfr=None,
                    attn_sfr_weights=None, attn_rcs_weights=None, use_custom_attn=False, guide_attn=None, generate_guide_attn=False):
        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if generate_guide_attn:
            attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale
            attn_weights = attn_weights /(attn_weights.max(dim=-1)[0].unsqueeze(-1))
            attn_weights = F.softmax(6 * attn_weights, dim=-1)

        elif guide_attn is not None:
            attn_weights = guide_attn.repeat(bsz, 1, 1)

        elif use_custom_attn:
            addition = self.addition_cache.get(n_patches)
            if addition is None:
                window_size = [side * 2 - 1 for side in n_patches]
                window = VisionTransformer.gaussian_window(*window_size, std=self.gaussian_std)
                addition = VisionTransformer.get_attention_addition(*n_patches, window).unsqueeze(0).to(x.dtype).to(
                    x.device)
                self.addition_cache[n_patches] = addition  # (1,197,197)
            omega = addition.clone()

            if self.attn_strategy == 'middleclip':
                attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale

                if attn_sfr is not None:
                    attn_sfr = attn_sfr / torch.max(attn_sfr)
                    attn_sfr = attn_sfr.to(torch.float16)

                    if attn_sfr_weights is not None:
                        tau_attn_sfr, lambda_attn_sfr = attn_sfr_weights
                    else:
                        raise NotImplemented
                    attn_weights = tau_attn_sfr * ((1 - lambda_attn_sfr) * attn_weights + lambda_attn_sfr * attn_sfr)

                attn_weights += omega
                attn_weights = F.softmax(attn_weights, dim=-1)

                if attn_rcs is not None:
                    if attn_rcs_weights is not None:
                        tau_attn_rcs, lambda_attn_rcs = attn_rcs_weights
                    else:
                        raise NotImplemented
                    attn_weights = tau_attn_rcs * ((1 - lambda_attn_rcs) * attn_weights + lambda_attn_rcs * attn_rcs)

            elif self.attn_strategy == 'naclip':
                attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale  

                if attn_sfr is not None:
                    attn_sfr = attn_sfr / torch.max(attn_sfr)
                    attn_sfr = attn_sfr.to(torch.float16)

                    if attn_sfr_weights is not None:
                        tau_attn_sfr, lambda_attn_sfr = attn_sfr_weights
                    else:
                        raise NotImplemented
                    attn_weights = tau_attn_sfr * ((1 - lambda_attn_sfr) * attn_weights + lambda_attn_sfr * attn_sfr)

                attn_weights += omega
                attn_weights = F.softmax(attn_weights, dim=-1)

                if attn_rcs is not None:
                    if attn_rcs_weights is not None:
                        tau_attn_rcs, lambda_attn_rcs = attn_rcs_weights
                    else:
                        raise NotImplemented
                    attn_weights = tau_attn_rcs * ((1 - lambda_attn_rcs) * attn_weights + lambda_attn_rcs * attn_rcs)

            elif self.attn_strategy == 'sclip':
                q_attn = torch.bmm(q, q.transpose(1, 2)) * scale
                k_attn = torch.bmm(k, k.transpose(1, 2)) * scale

                if attn_sfr is not None:
                    attn_sfr = attn_sfr / torch.max(attn_sfr)
                    attn_sfr = attn_sfr.to(torch.float16)

                    if attn_sfr_weights is not None:
                        tau_attn_sfr, lambda_attn_sfr = attn_sfr_weights
                    else:
                        raise NotImplemented
                    q_attn = tau_attn_sfr * ((1 - lambda_attn_sfr) * q_attn + lambda_attn_sfr * attn_sfr)
                    k_attn = tau_attn_sfr * ((1 - lambda_attn_sfr) * k_attn + lambda_attn_sfr * attn_sfr)

                attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)

                if attn_rcs is not None:
                    if attn_rcs_weights is not None:
                        tau_attn_rcs, lambda_attn_rcs = attn_rcs_weights
                    else:
                        raise NotImplemented
                    attn_weights = tau_attn_rcs * ((1 - lambda_attn_rcs) * attn_weights + lambda_attn_rcs * attn_rcs)

            elif self.attn_strategy == 'clearclip':
                attn_weights = torch.bmm(q, q.transpose(1, 2)) * scale

                if attn_sfr is not None:
                    attn_sfr = attn_sfr / torch.max(attn_sfr)
                    attn_sfr = attn_sfr.to(torch.float16)

                    if attn_sfr_weights is not None:
                        tau_attn_sfr, lambda_attn_sfr = attn_sfr_weights
                    else:
                        raise NotImplemented
                    attn_weights = tau_attn_sfr * ((1 - lambda_attn_sfr) * attn_weights + lambda_attn_sfr * attn_sfr)

                attn_weights = F.softmax(attn_weights, dim=-1)

                if attn_rcs is not None:
                    if attn_rcs_weights is not None:
                        tau_attn_rcs, lambda_attn_rcs = attn_rcs_weights
                    else:
                        raise NotImplemented
                    attn_weights = tau_attn_rcs * ((1 - lambda_attn_rcs) * attn_weights + lambda_attn_rcs * attn_rcs)

            elif self.attn_strategy == 'vanilla':
                attn_weights = torch.bmm(q * scale, k.transpose(1, 2))

                if attn_sfr is not None:
                    attn_sfr = attn_sfr / torch.max(attn_sfr)
                    attn_sfr = attn_sfr.to(torch.float16)

                    if attn_sfr_weights is not None:
                        tau_attn_sfr, lambda_attn_sfr = attn_sfr_weights
                    else:
                        raise NotImplemented
                    attn_weights = tau_attn_sfr * ((1 - lambda_attn_sfr) * attn_weights + lambda_attn_sfr * attn_sfr)

                attn_weights = F.softmax(attn_weights, dim=-1)

                if attn_rcs is not None:
                    if attn_rcs_weights is not None:
                        tau_attn_rcs, lambda_attn_rcs = attn_rcs_weights
                    else:
                        raise NotImplemented
                    attn_weights = tau_attn_rcs * ((1 - lambda_attn_rcs) * attn_weights + lambda_attn_rcs * attn_rcs)

            else:
                raise NotImplemented(f'attn_strategy {self.attn_strategy} is not implemented')

        else:  
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)

        if return_attn:
            return attn_weights

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)

        if with_attn:
            return attn_output, attn_weights

        return attn_output



class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,  # 512
                 # vision
                 image_resolution: int,  # 224
                 vision_layers: Union[Tuple[int, int, int, int], int],  # 12
                 vision_width: int,  # 768
                 vision_patch_size: int,  # 16
                 # text
                 context_length: int,  # 77
                 vocab_size: int,  # 49408
                 transformer_width: int,  # 512
                 transformer_heads: int,  # 8
                 transformer_layers: int  # 12
                 ):
        super().__init__()
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_all=False, use_rcs=True, use_sfr=True,
                     temp_thd=None, attn_sfr_weights=None, attn_rcs_weights=None, delete_same_entity=False,
                     query_features=None, img=None, patch_size=None, logit_scale=None, dtype=None, align_corners=None):
        return self.visual(image.type(self.dtype), return_all=return_all, use_rcs=use_rcs, use_sfr=use_sfr,
                           temp_thd=temp_thd, attn_sfr_weights=attn_sfr_weights, attn_rcs_weights=attn_rcs_weights,
                           delete_same_entity=delete_same_entity, query_features=query_features, img=img,
                           patch_size=patch_size, logit_scale=logit_scale, dtype=dtype, align_corners=align_corners)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
