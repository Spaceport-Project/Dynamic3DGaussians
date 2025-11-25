
# import io
import os
import random
import threading
from typing import Dict
import torch
import torch.nn.functional as F  

import numpy as np
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import torch.utils
import torchvision
from helpers import setup_camera, quat_mult, searchForMaxIteration
from external import build_rotation
from colormap import colormap
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
import sys
from PIL import Image
import viser
from viser import transforms as tf
import signal
from encoders import gst_h264_endoder_pipeline
import ctypes
from ctypes import string_at
from ctypes import *
from scipy.io.wavfile import write
from pydub import AudioSegment
from io import BytesIO  




libc = CDLL("libc.so.6") 
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstVideo', '1.0')
# gi.require_version('GstCuda', '1.0')
from gi.repository import Gst, GstApp, GLib, GObject
Gst.init(None)


REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

w, h = 1920, 1080
near, far = 0.01, 100.0

# def_pix = torch.tensor(
#     np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
# pix_ones = torch.ones(h * w, 1).cuda().float()

# to8b = lambda x : (255*np.clip(x.permute(1,2,0).contiguous().cpu().detach().numpy(),0,1)).astype(np.uint8)

import torch
import torch.nn.functional as F

import kornia


import torch

class HSLGrayColorReplacer:
    def __init__(self, device='cuda'):
        self.device = device

    def rgb_to_hsl(self, rgb_tensor):
        """Convert RGB tensor to HSL - fixed version"""
        r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
        
        # Find min and max values
        rgb_stack = torch.stack([r, g, b], dim=0)
        max_val, max_idx = torch.max(rgb_stack, dim=0)
        min_val, _ = torch.min(rgb_stack, dim=0)
        diff = max_val - min_val
        
        # Lightness
        l = (max_val + min_val) / 2.0
        
        # Saturation
        s = torch.zeros_like(l)
        non_zero_diff = diff > 1e-6
        
        # When L <= 0.5
        mask_low = non_zero_diff & (l <= 0.5)
        s[mask_low] = diff[mask_low] / (max_val[mask_low] + min_val[mask_low])
        
        # When L > 0.5  
        mask_high = non_zero_diff & (l > 0.5)
        s[mask_high] = diff[mask_high] / (2.0 - max_val[mask_high] - min_val[mask_high])
        
        # Hue calculation
        h = torch.zeros_like(l)
        
        # Red is max (max_idx == 0)
        red_max = (max_idx == 0) & non_zero_diff
        h[red_max] = ((g[red_max] - b[red_max]) / diff[red_max]) % 6
        
        # Green is max (max_idx == 1)
        green_max = (max_idx == 1) & non_zero_diff
        h[green_max] = (b[green_max] - r[green_max]) / diff[green_max] + 2
        
        # Blue is max (max_idx == 2)
        blue_max = (max_idx == 2) & non_zero_diff
        h[blue_max] = (r[blue_max] - g[blue_max]) / diff[blue_max] + 4
        
        # Convert to degrees and normalize
        h = (h * 60) / 360.0  # Normalize to [0, 1]
        h = torch.where(h < 0, h + 1.0, h)  # Handle negative values
        
        return torch.stack([h, s, l])  # Make sure this line executes

    def replace_gray_with_color(self, frame_tensor, target_rgb_color):
        """Replace gray colors using mask"""
        # Ensure (3, H, W) format and float
        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.float() / 255.0

        # Convert to HSL for gray detection
        hsl_tensor = self.rgb_to_hsl(frame_tensor)
        
        # Add debug check
        if hsl_tensor is None:
            print("Error: HSL conversion returned None")
            return frame_tensor
            
        H, S, L = hsl_tensor[0], hsl_tensor[1], hsl_tensor[2]

        # Define gray range
        lower_lightness = 80/255.0
        upper_lightness = 160/255.0
        max_saturation = 0.05

        # Gray mask
        is_gray = (S <= max_saturation) & (L >= lower_lightness) & (L <= upper_lightness)

        # Create target color tensor
        target_tensor = torch.zeros_like(frame_tensor)
        target_tensor[0] = target_rgb_color[0]  # R
        target_tensor[1] = target_rgb_color[1]  # G  
        target_tensor[2] = target_rgb_color[2]  # B

        # Use mask to replace
        result = torch.where(is_gray.unsqueeze(0), target_tensor, frame_tensor)

        return (result * 255.0).clamp(0, 255).to(torch.uint8)
class TemporalHSLStabilizer:
    def __init__(self, window_size=5, device='cuda'):
        self.device = device
        self.frame_buffer = []
        self.window_size = window_size
        self.prev_frame = None

    def rgb_to_hsl(self, rgb_tensor):
        """Convert RGB tensor to HSL - optimized version"""
        r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
        
        # Find min and max values
        rgb_stack = torch.stack([r, g, b], dim=0)
        max_val, max_idx = torch.max(rgb_stack, dim=0)
        min_val, _ = torch.min(rgb_stack, dim=0)
        diff = max_val - min_val
        
        # Lightness
        l = (max_val + min_val) / 2.0
        
        # Saturation
        s = torch.zeros_like(l)
        non_zero_diff = diff > 1e-6
        
        # When L <= 0.5
        mask_low = non_zero_diff & (l <= 0.5)
        s[mask_low] = diff[mask_low] / (max_val[mask_low] + min_val[mask_low])
        
        # When L > 0.5
        mask_high = non_zero_diff & (l > 0.5)
        s[mask_high] = diff[mask_high] / (2.0 - max_val[mask_high] - min_val[mask_high])
        
        # Hue calculation
        h = torch.zeros_like(l)
        
        # Red is max (max_idx == 0)
        red_max = (max_idx == 0) & non_zero_diff
        h[red_max] = ((g[red_max] - b[red_max]) / diff[red_max]) % 6
        
        # Green is max (max_idx == 1)
        green_max = (max_idx == 1) & non_zero_diff
        h[green_max] = (b[green_max] - r[green_max]) / diff[green_max] + 2
        
        # Blue is max (max_idx == 2)
        blue_max = (max_idx == 2) & non_zero_diff
        h[blue_max] = (r[blue_max] - g[blue_max]) / diff[blue_max] + 4
        
        # Convert to degrees and normalize
        h = (h * 60) / 360.0  # Normalize to [0, 1]
        h[h < 0] += 1.0
        
        return torch.stack([h, s, l])

    def process_frame(self, frame_tensor):
        # Ensure (3, H, W) format and float
        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.float() / 255.0

        # Convert RGB to HSL
        hsl_tensor = self.rgb_to_hsl(frame_tensor)
        H, S, L = hsl_tensor[0], hsl_tensor[1], hsl_tensor[2]

        # Define gray range in HSL space
        # Gray colors have very low saturation (0-10%) and lightness between ~31-63%
        lower_lightness = 80/255.0  # ~31%
        upper_lightness = 160/255.0  # ~63%
        max_saturation = 0.1  # 10% saturation threshold for gray

        # Gray mask: low saturation AND lightness in range
        is_gray = (S <= max_saturation) & (L >= lower_lightness) & (L <= upper_lightness)
        gray_mask = is_gray.float()

        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            self.frame_buffer.append(frame_tensor.clone())
            if len(self.frame_buffer) > self.window_size:
                self.frame_buffer.pop(0)
            return (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)

        self.frame_buffer.append(frame_tensor.clone())
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

        # Weighted average with more weight on recent frames
        weights = torch.linspace(0.1, 1.0, len(self.frame_buffer))
        weights = weights / weights.sum()

        result = torch.zeros_like(frame_tensor)
        for i, frame in enumerate(self.frame_buffer):
            result += frame * weights[i]

        # Blend gray regions with previous frame (50%)
        result = frame_tensor * (1 - gray_mask * 0.5) + self.prev_frame * (gray_mask * 0.5)
        self.prev_frame = frame_tensor.clone()

        return (result * 255.0).clamp(0, 255).to(torch.uint8)

class TemporalRGBStabilizer:
    def __init__(self, window_size=5, device='cuda'):
        self.device = device
        self.frame_buffer = []
        self.window_size = window_size
        self.prev_frame = None  # Store previous frame for gray mask blending

    def process_frame(self, frame_tensor):
        # Ensure (3, H, W) format and float
        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.float() / 255.0

        # Extract RGB channels
        R, G, B = frame_tensor[0], frame_tensor[1], frame_tensor[2]

        # Define gray color range
        lower_gray = torch.tensor([80/255.0, 80/255.0, 80/255.0]).to(frame_tensor.device)
        upper_gray = torch.tensor([160/255.0, 160/255.0, 160/255.0]).to(frame_tensor.device)

        # Check if each channel falls within the gray range
        is_gray = (R >= lower_gray[0]) & (R <= upper_gray[0]) & \
                  (G >= lower_gray[1]) & (G <= upper_gray[1]) & \
                  (B >= lower_gray[2]) & (B <= upper_gray[2])

        gray_mask = is_gray.float()

        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            self.frame_buffer.append(frame_tensor.clone())
            if len(self.frame_buffer) > self.window_size:
                self.frame_buffer.pop(0)
            return (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)

        self.frame_buffer.append(frame_tensor.clone())
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

        # Weighted average with more weight on recent frames
        weights = torch.linspace(0.1, 1.0, len(self.frame_buffer))
        weights = weights / weights.sum()

        result = torch.zeros_like(frame_tensor)
        for i, frame in enumerate(self.frame_buffer):
            result += frame * weights[i]

        # Blend gray regions with previous frame (50%)
        result = frame_tensor * (1 - gray_mask * 0.5) + self.prev_frame * (gray_mask * 0.5)
        self.prev_frame = frame_tensor.clone()

        return (result * 255.0).clamp(0, 255).to(torch.uint8)

class TemporalLABStabilizer:
    def __init__(self, window_size=5, device='cuda'):
        self.device = device
        self.frame_buffer = []
        self.window_size = window_size
        self.prev_frame = None  # Store previous frame for gray mask blending

    def process_frame(self, frame_tensor):
        # Ensure (3, H, W) format and float
        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.float() / 255.0

        # Kornia expects a batch dimension: (B, 3, H, W)
        frame_batched = frame_tensor.unsqueeze(0)

        # Convert to LAB
        lab = kornia.color.rgb_to_lab(frame_batched)  # (1, 3, H, W)
        L, a, b = lab[0, 0], lab[0, 1], lab[0, 2]

        # Gray: low chroma (a and b near zero), L in 40-160
        is_gray = (torch.abs(a) < 8) & (torch.abs(b) < 8) & (L >= 30.0) & (L <= 80.0)
        gray_mask = is_gray.float()

        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            self.frame_buffer.append(frame_tensor.clone())
            if len(self.frame_buffer) > self.window_size:
                self.frame_buffer.pop(0)
            return (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)

        self.frame_buffer.append(frame_tensor.clone())
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

        # Weighted average with more weight on recent frames
        weights = torch.linspace(0.1, 1.0, len(self.frame_buffer))
        weights = weights / weights.sum()

        result = torch.zeros_like(frame_tensor)
        for i, frame in enumerate(self.frame_buffer):
            result += frame * weights[i]

        # Blend gray regions with previous frame (50%)
        result = frame_tensor * (1 - gray_mask * 0.5) + self.prev_frame * (gray_mask * 0.5)
        self.prev_frame = frame_tensor.clone()

        return (result * 255.0).clamp(0, 255).to(torch.uint8)

class ColorizeGrayRegionsLAB:
    def __init__(self, color=(130./255, 130.0/255, 130.0/255), device='cuda'):
        """
        Colorize gray regions (LAB) with low chroma and L in 40-160 with a specified color.

        Args:
            color: RGB tuple (r, g, b) in the range of [0, 1]
            device: GPU device
        """
        self.device = device
        self.prev_frame = None
        self.target_color = torch.tensor(color, device=self.device).view(3, 1, 1)

        # Define the gray range
        self.gray_min = 30 #40.0
        self.gray_max = 60 #160.0
        self.chroma_threshold = 8.0

    def process_frame(self, frame_tensor):
        # Ensure (3, H, W) format and float
        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.float() / 255.0

        # Kornia expects a batch dimension: (B, 3, H, W)
        frame_batched = frame_tensor.unsqueeze(0)

        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            return (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)

        # Convert to LAB
        lab = kornia.color.rgb_to_lab(frame_batched)  # (1, 3, H, W)
        L, a, b = lab[0, 0], lab[0, 1], lab[0, 2]

        # Gray: low chroma (a and b near zero), L in 40-160
        is_gray = (torch.abs(a) < self.chroma_threshold) & \
                  (torch.abs(b) < self.chroma_threshold) & \
                  (L >= self.gray_min) & (L <= self.gray_max)
        gray_mask = is_gray.float()

        # Colorize gray regions with the target color
        result = frame_tensor * (1 - gray_mask) + self.target_color * gray_mask

        self.prev_frame = frame_tensor.clone()
        return (result * 255.0).clamp(0, 255).to(torch.uint8)

class ColorizeGrayRegions:
    def __init__(self, color=(0.0, 0.0, 1.0), device='cuda'):
        """
        Colorize gray regions in the range of 40-160 RGB with a specified color.

        Args:
            color: RGB tuple (r, g, b) in the range of [0, 1]
            device: GPU device
        """
        self.device = device
        self.prev_frame = None
        self.target_color = torch.tensor(color, device=self.device).view(3, 1, 1)

        # Define the gray range
        self.gray_min = 60.0 / 255.0
        self.gray_max = 160.0 / 255.0
        self.gray_tolerance = 10.0 / 255.0

    def process_frame(self, frame_tensor):
        # Ensure (3, H, W) format and float
        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.float() / 255.0

        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            return (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)

        r, g, b = frame_tensor[0], frame_tensor[1], frame_tensor[2]

        # Calculate average RGB value for each pixel
        avg_rgb = (r + g + b) / 3.0

        # Check if pixel is in gray range (40-160)
        in_gray_range = (avg_rgb >= self.gray_min) & (avg_rgb <= self.gray_max)

        # Check if RGB values are close to each other (neutral gray)
        max_channel = torch.max(torch.max(r, g), b)
        min_channel = torch.min(torch.min(r, g), b)
        is_neutral = (max_channel - min_channel) < self.gray_tolerance

        # Combine conditions: must be in gray range AND neutral
        gray_mask = (in_gray_range & is_neutral).float()

        # Colorize gray regions with the target color
        result = frame_tensor * (1 - gray_mask) + self.target_color * gray_mask

        self.prev_frame = frame_tensor.clone()
        return (result * 255.0).clamp(0, 255).to(torch.uint8)
class SimpleLABGrayStabilizer:
    def __init__(self, device='cuda'):
        self.device = device
        self.prev_frame = None

    def process_frame(self, frame_tensor):
        # Ensure (3, H, W) format and float
        # frame_tensor = reduce_color_depth_with_dithering(frame_tensor, bits=6)  


        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        
        
        frame_tensor = frame_tensor.float() / 255.0

        # Kornia expects a batch dimension: (B, 3, H, W)
        frame_batched = frame_tensor.unsqueeze(0)

        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            return (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)

        # Convert to LAB
        lab = kornia.color.rgb_to_lab(frame_batched)  # (1, 3, H, W)
        L, a, b = lab[0, 0], lab[0, 1], lab[0, 2]

        # Gray: low chroma (a and b near zero), L in 40-160
        is_gray = (torch.abs(a) < 8) & (torch.abs(b) < 8) & (L >= 30.0) & (L <= 80.0)
        gray_mask = is_gray.float()

        # Blend gray regions with previous frame (50%)
        result = frame_tensor * (1 - gray_mask * 0.5) + self.prev_frame * (gray_mask * 0.5)
        self.prev_frame = frame_tensor.clone()
        return (result * 255.0).clamp(0, 255).to(torch.uint8)
    
class SimpleHSVGrayStabilizer:
    def __init__(self, device='cuda'):
        self.device = device
        self.prev_frame = None

    def process_frame(self, frame_tensor):
        # Ensure (3, H, W) format and float
        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.float() / 255.0

        # Kornia expects a batch dimension: (B, 3, H, W)
        frame_batched = frame_tensor.unsqueeze(0)

        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            return (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)

        # Convert to HSV
        hsv = kornia.color.rgb_to_hsv(frame_batched)  # (1, 3, H, W)
        h, s, v = hsv[0, 0], hsv[0, 1], hsv[0, 2]

        # Gray: low saturation, value in 40-160
        is_gray = (s < 0.15) & (v >= 40.0/255.0) & (v <= 160.0/255.0)
        gray_mask = is_gray.float()

        # Blend gray regions with previous frame (50%)
        result = frame_tensor * (1 - gray_mask * 0.5) + self.prev_frame * (gray_mask * 0.5)
        self.prev_frame = frame_tensor.clone()
        return (result * 255.0).clamp(0, 255).to(torch.uint8)
class GrayTshirtFlickerReducer:
    def __init__(self, device='cuda'):
        """
        Specifically target gray t-shirt flicker based on RGB values 80-160
        """
        self.device = device
        self.prev_frame = None
        
        # Define the gray t-shirt range - expanded to 80-160
        self.gray_min = 80.0 / 255.0   # Darkest gray region
        self.gray_max = 160.0 / 255.0  # Brightest gray region
        self.gray_tolerance = 10.0 / 255.0  # Allow some variation in RGB channels
        
    def process_frame(self, frame_tensor):
        # Store original shape and format
        original_shape = frame_tensor.shape
        original_dtype = frame_tensor.dtype
        
        # Convert to float [0, 1]
        frame_tensor = frame_tensor.float() / 255.0
        
        # Handle different input formats
        if len(frame_tensor.shape) == 3:
            if frame_tensor.shape[0] == 3:  # Already (3, H, W)
                needs_permute_back = False
            elif frame_tensor.shape[2] == 3:  # (H, W, 3) format
                frame_tensor = frame_tensor.permute(2, 0, 1)  # Convert to (3, H, W)
                needs_permute_back = True
            else:
                raise ValueError(f"Unexpected tensor shape: {frame_tensor.shape}")
        else:
            raise ValueError(f"Expected 3D tensor, got shape: {frame_tensor.shape}")
        
        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            # Convert back to original format
            result = frame_tensor
            if needs_permute_back:
                result = result.permute(1, 2, 0)
            return (result * 255.0).clamp(0, 255).to(original_dtype)
        
        r, g, b = frame_tensor[0], frame_tensor[1], frame_tensor[2]
        
        # Calculate average RGB value for each pixel
        avg_rgb = (r + g + b) / 3.0
        
        # Check if pixel is in gray t-shirt range (80-160)
        in_gray_range = (avg_rgb >= self.gray_min) & (avg_rgb <= self.gray_max)
        
        # Check if RGB values are close to each other (neutral gray)
        max_channel = torch.max(torch.max(r, g), b)
        min_channel = torch.min(torch.min(r, g), b)
        is_neutral = (max_channel - min_channel) < self.gray_tolerance
        
        # Combine conditions: must be in gray range AND neutral
        gray_tshirt_mask = (in_gray_range & is_neutral).float()
        
        # For gray t-shirt regions, quantize more aggressively
        # Quantize to fewer levels (like 4-bit = 16 levels)
        quantization_levels = 15  # 4-bit quantization
        
        # Quantize each channel
        quantized_r = torch.round(r * quantization_levels) / quantization_levels
        quantized_g = torch.round(g * quantization_levels) / quantization_levels
        quantized_b = torch.round(b * quantization_levels) / quantization_levels
        
        # Apply quantization only to gray t-shirt regions
        result_r = r * (1 - gray_tshirt_mask) + quantized_r * gray_tshirt_mask
        result_g = g * (1 - gray_tshirt_mask) + quantized_g * gray_tshirt_mask
        result_b = b * (1 - gray_tshirt_mask) + quantized_b * gray_tshirt_mask
        
        result = torch.stack([result_r, result_g, result_b], dim=0)
        
        self.prev_frame = frame_tensor.clone()
        
        # Convert back to original format
        if needs_permute_back:
            result = result.permute(1, 2, 0)  # Convert back to (H, W, 3)
        
        return (result * 255.0).clamp(0, 255).to(original_dtype)

class GrayTshirtTemporalStabilizer:
    def __init__(self, device='cuda'):
        """
        Use temporal averaging specifically for gray t-shirt regions (80-160)
        """
        self.device = device
        self.prev_frame = None
        self.running_average = None
        
        # Gray t-shirt specific parameters - expanded range
        self.gray_min = 75.0 / 255.0   # Slightly wider range (75-165)
        self.gray_max = 165.0 / 255.0  # Slightly wider range
        self.gray_tolerance = 15.0 / 255.0
        
    def process_frame(self, frame_tensor):
        # Convert to float [0, 1]
        frame_tensor = frame_tensor.float() / 255.0
        
        # Ensure (3, H, W) format
        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        
        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            self.running_average = frame_tensor.clone()
            return (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        
        r, g, b = frame_tensor[0], frame_tensor[1], frame_tensor[2]
        
        # Detect gray t-shirt regions (80-160 range)
        avg_rgb = (r + g + b) / 3.0
        in_gray_range = (avg_rgb >= self.gray_min) & (avg_rgb <= self.gray_max)
        
        max_channel = torch.max(torch.max(r, g), b)
        min_channel = torch.min(torch.min(r, g), b)
        is_neutral = (max_channel - min_channel) < self.gray_tolerance
        
        gray_tshirt_mask = (in_gray_range & is_neutral).float()
        
        # For gray t-shirt regions, use temporal averaging
        alpha = 0.3  # Keep 30% of current frame, 70% of running average
        
        # Update running average for gray regions
        self.running_average = self.running_average * (1 - gray_tshirt_mask * alpha) + \
                              frame_tensor * (gray_tshirt_mask * alpha) + \
                              frame_tensor * (1 - gray_tshirt_mask)
        
        # Use running average for gray regions, current frame for others
        result = frame_tensor * (1 - gray_tshirt_mask) + self.running_average * gray_tshirt_mask
        
        self.prev_frame = frame_tensor.clone()
        
        return (result * 255.0).clamp(0, 255).to(torch.uint8)

class SimpleGrayStabilizer:
    def __init__(self, device='cuda', blend_factor = 0.2):
        self.device = device
        self.prev_frame = None
        self.blend_factor = blend_factor
        
    def process_frame(self, frame_tensor):
        # frame_tensor = reduce_color_depth_with_dithering(frame_tensor, bits=5)  
        frame_tensor = frame_tensor.float() / 255.0
        
        if frame_tensor.shape[0] != 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        
        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            return (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        
        r, g, b = frame_tensor[0], frame_tensor[1], frame_tensor[2]
        
        # Gray detection: RGB values between 80-160 and similar to each other
        avg_rgb = (r + g + b) / 3.0
        is_gray_range = (avg_rgb >= 40.0/255.0) & (avg_rgb <= 160.0/255.0)
        
        # Check if R, G, B are similar (difference < 10)
        max_diff = torch.max(torch.max(torch.abs(r-g), torch.abs(g-b)), torch.abs(r-b))
        is_similar_rgb = max_diff < 10.0/255.0
        
        gray_mask = (is_gray_range & is_similar_rgb).float()
        
        # For gray regions, blend 50% with previous frame
        result = frame_tensor * (1 - gray_mask * self.blend_factor) + self.prev_frame * (gray_mask * self.blend_factor)
        
        self.prev_frame = frame_tensor.clone()
        
        return (result * 255.0).clamp(0, 255).to(torch.uint8)

class LuminanceQuantizer:
    def __init__(self, luma_bits=4, device='cuda'):
        """
        Quantize only the luminance component, leave chrominance alone
        """
        self.luma_bits = luma_bits
        self.device = device
        
    def process_frame(self, frame_tensor):
        # Debug: Print input shape
        print(f"Input frame_tensor shape: {frame_tensor.shape}")
        
        # Handle different input formats
        if len(frame_tensor.shape) == 3:
            if frame_tensor.shape[0] == 3:  # (3, H, W) format - RGB channels first
                C, H, W = frame_tensor.shape
                is_chw_format = True
            elif frame_tensor.shape[2] == 3:  # (H, W, 3) format - RGB channels last
                H, W, C = frame_tensor.shape
                frame_tensor = frame_tensor.permute(2, 0, 1)  # Convert to (3, H, W)
                C, H, W = frame_tensor.shape
                is_chw_format = False
            else:
                raise ValueError(f"Unexpected tensor shape: {frame_tensor.shape}")
        else:
            raise ValueError(f"Expected 3D tensor, got shape: {frame_tensor.shape}")
        
        print(f"After format check - C: {C}, H: {H}, W: {W}")
        
        frame_tensor = frame_tensor.float() / 255.0
        
        # Calculate luminance - now we know frame_tensor is (3, H, W)
        luma_weights = torch.tensor([0.299, 0.587, 0.114], device=self.device).view(3, 1, 1)
        luminance = torch.sum(frame_tensor * luma_weights, dim=0, keepdim=True)  # (1, H, W)
        
        # print(f"Luminance shape: {luminance.shape}")
        
        # Quantize luminance
        luma_levels = 2 ** self.luma_bits - 1
        quantized_luma = torch.round(luminance * luma_levels) / luma_levels
        
        # Apply the luminance change proportionally to all channels
        luma_ratio = quantized_luma / (luminance + 1e-8)  # Avoid division by zero
        quantized_frame = frame_tensor * luma_ratio
        
        # Convert back to original format if needed
        if not is_chw_format:
            quantized_frame = quantized_frame.permute(1, 2, 0)  # Convert back to (H, W, 3)
        
        result = (quantized_frame * 255.0).clamp(0, 255).to(torch.uint8)
        # print(f"Output result shape: {result.shape}")
        
        return result
def reduce_color_depth_with_dithering(frame_tensor, bits=6):
    """
    Reduce the color depth of a PyTorch tensor with dithering.

    Args:
        frame_tensor: PyTorch tensor of shape (C, H, W) or (H, W, C)
        bits: Number of bits to use for each color channel (e.g., 6, 4, 2)

    Returns:
        PyTorch tensor with reduced color depth and dithering
    """
    # Convert to float and scale to [0, 1]
    frame_tensor = frame_tensor.float() / 255.0
    # torchvision.utils.save_image(frame_tensor.permute(2,0,1), "im_org.png")
    # Quantize the tensor
    levels = 2 ** bits - 1
    quantized_tensor = torch.floor(frame_tensor * levels) / levels

    # Add dithering noise
    noise = torch.rand_like(frame_tensor) / levels
    dithered_tensor = quantized_tensor + noise
    # torchvision.utils.save_image(dithered_tensor.permute(2,0,1), "im.png")

    # Convert back to uint8 and scale to [0, 255]
    dithered_tensor = (dithered_tensor * 255.0).clamp(0, 255).to(torch.uint8)

    return dithered_tensor

class ImprovedExponentialMovingAverageSmoother:
    def __init__(self, alpha=0.7, flicker_threshold=0.05, adaptive_alpha=True, device='cuda'):
        """
        Improved exponential moving average smoother
        
        Args:
            alpha: Base smoothing factor (0.0 = all smoothing, 1.0 = no smoothing)
            flicker_threshold: Threshold for detecting flicker
            adaptive_alpha: Whether to adapt alpha based on flicker amount
            device: GPU device
        """
        self.alpha_base = alpha
        self.flicker_threshold = flicker_threshold
        self.adaptive_alpha = adaptive_alpha
        self.device = device
        self.running_average = None
        self.last_frame = None

    def process_frame(self, frame_tensor):
        # Convert to float and scale to [0, 1]
        frame_tensor = frame_tensor.float() / 255.0

        if self.running_average is None:
            self.running_average = frame_tensor.clone()
            self.last_frame = frame_tensor.clone()
            result = (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)
            return result

        # Calculate flicker amount (more sophisticated)
        if self.last_frame is not None:
            # Calculate per-pixel difference
            diff = torch.abs(frame_tensor - self.last_frame)
            
            # Weight by luminance (brighter pixels get more weight in flicker calculation)
            if frame_tensor.shape[0] == 3:  # RGB
                luminance = 0.299 * frame_tensor[0] + 0.587 * frame_tensor[1] + 0.114 * frame_tensor[2]
                weight = 1.0 + 2.0 * luminance  # Bright pixels get 3x weight
                weighted_diff = diff * weight.unsqueeze(0)
                flicker = torch.mean(weighted_diff).item()
            else:
                flicker = torch.mean(diff).item()
        else:
            flicker = 0.0
        # print(flicker)

        # Adaptive alpha based on flicker amount
        if self.adaptive_alpha and flicker > self.flicker_threshold:
            # More flicker = lower alpha (more smoothing)
            alpha = max(0.3, self.alpha_base - (flicker - self.flicker_threshold) * 2.0)
        else:
            alpha = self.alpha_base

        # Apply smoothing only if flicker is detected
        if flicker > self.flicker_threshold:
            self.running_average = alpha * frame_tensor + (1 - alpha) * self.running_average
            result = self.running_average
        else:
            # No significant flicker, use original frame
            self.running_average = frame_tensor.clone()
            result = frame_tensor

        self.last_frame = frame_tensor.clone()

        # Convert back to uint8 and scale to [0, 255]
        result = (result * 255.0).clamp(0, 255).to(torch.uint8)
        return result

    def reset(self):
        self.running_average = None
        self.last_frame = None
class YUVExponentialMovingAverageSmoother:
    def __init__(self, alpha_y=0.8, alpha_uv=0.5, flicker_threshold=0.1, device='cuda'):
        self.alpha_y = alpha_y
        self.alpha_uv = alpha_uv
        self.flicker_threshold = flicker_threshold
        self.device = device
        self.running_average = None
        self.last_frame = None

        # YUV conversion matrix (ITU-R BT.601)
        self.yuv_weights = torch.tensor([
            [0.299,    0.587,    0.114   ],
            [-0.14713, -0.28886, 0.436   ],
            [0.615,   -0.51499, -0.10001 ]
        ], device=self.device, dtype=torch.float32)
        self.inv_yuv_weights = torch.linalg.inv(self.yuv_weights)

    def rgb_to_yuv(self, rgb_frame):
        # Debug: Print the input shape
        print(f"Input rgb_frame shape: {rgb_frame.shape}")
        
        # Handle different input formats
        if len(rgb_frame.shape) == 3:
            if rgb_frame.shape[0] == 3:  # (3, H, W) format
                C, H, W = rgb_frame.shape
            elif rgb_frame.shape[2] == 3:  # (H, W, 3) format
                H, W, C = rgb_frame.shape
                rgb_frame = rgb_frame.permute(2, 0, 1)  # Convert to (3, H, W)
                C, H, W = rgb_frame.shape
            else:
                raise ValueError(f"Unexpected tensor shape: {rgb_frame.shape}")
        else:
            raise ValueError(f"Expected 3D tensor, got shape: {rgb_frame.shape}")
        
        print(f"After format check - C: {C}, H: {H}, W: {W}")
        
        # Convert to (H*W, 3) for matrix multiplication
        pixels = rgb_frame.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
        print(f"Pixels shape after reshape: {pixels.shape}")
        
        # Matrix multiplication
        yuv_pixels = torch.matmul(pixels, self.yuv_weights.T)  # (H*W, 3)
        print(f"YUV pixels shape after matmul: {yuv_pixels.shape}")
        
        # Reshape back to (3, H, W)
        yuv_frame = yuv_pixels.reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        print(f"Final YUV frame shape: {yuv_frame.shape}")
        
        return yuv_frame

    def yuv_to_rgb(self, yuv_frame):
        C, H, W = yuv_frame.shape
        pixels = yuv_frame.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
        rgb_pixels = torch.matmul(pixels, self.inv_yuv_weights.T)  # (H*W, 3)
        rgb_frame = rgb_pixels.reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        return rgb_frame

    def process_frame(self, frame_tensor):
        print(f"Input frame_tensor shape: {frame_tensor.shape}")
        
        frame_tensor = frame_tensor.float() / 255.0
        yuv_frame = self.rgb_to_yuv(frame_tensor)

        if self.running_average is None:
            self.running_average = yuv_frame.clone()
            self.last_frame = yuv_frame.clone()
            rgb_frame = self.yuv_to_rgb(yuv_frame)
            rgb_frame = (rgb_frame * 255.0).clamp(0, 255).to(torch.uint8)
            return rgb_frame

        flicker = torch.mean(torch.abs(yuv_frame - self.last_frame)).item() if self.last_frame is not None else 0.0

        if flicker > self.flicker_threshold:
            self.running_average[0] = self.alpha_y * yuv_frame[0] + (1 - self.alpha_y) * self.running_average[0]  # Y
            self.running_average[1:] = self.alpha_uv * yuv_frame[1:] + (1 - self.alpha_uv) * self.running_average[1:]  # UV
            result = self.running_average
        else:
            self.running_average = yuv_frame.clone()
            result = yuv_frame

        self.last_frame = yuv_frame.clone()
        rgb_frame = self.yuv_to_rgb(result)
        rgb_frame = (rgb_frame * 255.0).clamp(0, 255).to(torch.uint8)
        return rgb_frame

    def reset(self):
        self.running_average = None
        self.last_frame = None
    def __init__(self, alpha_y=0.8, alpha_uv=0.5, flicker_threshold=0.1, device='cuda'):
        self.alpha_y = alpha_y
        self.alpha_uv = alpha_uv
        self.flicker_threshold = flicker_threshold
        self.device = device
        self.running_average = None
        self.last_frame = None

        # YUV conversion matrix (ITU-R BT.601)
        self.yuv_weights = torch.tensor([
            [0.299,    0.587,    0.114   ],
            [-0.14713, -0.28886, 0.436   ],
            [0.615,   -0.51499, -0.10001 ]
        ], device=self.device, dtype=torch.float32)
        self.inv_yuv_weights = torch.linalg.inv(self.yuv_weights)

    def rgb_to_yuv(self, rgb_frame):
        # rgb_frame: (3, H, W)
        C, H, W = rgb_frame.shape
        pixels = rgb_frame.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
        yuv_pixels = torch.matmul(pixels, self.yuv_weights.T)  # (H*W, 3)
        yuv_frame = yuv_pixels.reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        return yuv_frame

    def yuv_to_rgb(self, yuv_frame):
        # yuv_frame: (3, H, W)
        C, H, W = yuv_frame.shape
        pixels = yuv_frame.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
        rgb_pixels = torch.matmul(pixels, self.inv_yuv_weights.T)  # (H*W, 3)
        rgb_frame = rgb_pixels.reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)
        return rgb_frame

    def process_frame(self, frame_tensor):
        frame_tensor = frame_tensor.float() / 255.0
        yuv_frame = self.rgb_to_yuv(frame_tensor)

        if self.running_average is None:
            self.running_average = yuv_frame.clone()
            self.last_frame = yuv_frame.clone()
            rgb_frame = self.yuv_to_rgb(yuv_frame)
            rgb_frame = (rgb_frame * 255.0).clamp(0, 255).to(torch.uint8)
            return rgb_frame

        flicker = torch.mean(torch.abs(yuv_frame - self.last_frame)).item() if self.last_frame is not None else 0.0

        if flicker > self.flicker_threshold:
            self.running_average[0] = self.alpha_y * yuv_frame[0] + (1 - self.alpha_y) * self.running_average[0]  # Y
            self.running_average[1:] = self.alpha_uv * yuv_frame[1:] + (1 - self.alpha_uv) * self.running_average[1:]  # UV
            result = self.running_average
        else:
            self.running_average = yuv_frame.clone()
            result = yuv_frame

        self.last_frame = yuv_frame.clone()
        rgb_frame = self.yuv_to_rgb(result)
        rgb_frame = (rgb_frame * 255.0).clamp(0, 255).to(torch.uint8)
        return rgb_frame

    def reset(self):
        self.running_average = None
        self.last_frame = None

class GradientClippedExponentialMovingAverageSmoother:
    def __init__(self, alpha=0.7, flicker_threshold=0.1, gradient_clip=0.1, device='cuda'):
        """
        Exponential moving average smoother with temporal gradient clipping

        Args:
            alpha: Smoothing factor
            flicker_threshold: Apply smoothing only when flicker is detected
            gradient_clip: Maximum change allowed between frames
            device: GPU device
        """
        self.alpha = alpha
        self.flicker_threshold = flicker_threshold
        self.gradient_clip = gradient_clip
        self.device = device
        self.running_average = None
        self.last_frame = None

    def process_frame(self, frame_tensor):
        """
        Process frame with exponential moving average and gradient clipping
        """
        # Convert to float and scale to [0, 1]
        frame_tensor = frame_tensor.float() / 255.0

        if self.running_average is None:
            self.running_average = frame_tensor.clone()
            self.last_frame = frame_tensor.clone()
            # Convert back to uint8 and scale to [0, 255]
            result = (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)
            return result

        # Calculate flicker amount
        if self.last_frame is not None:
            flicker = torch.mean(torch.abs(frame_tensor - self.last_frame)).item()
        else:
            flicker = 0.0

        # Apply smoothing only if flicker is detected
        if flicker > self.flicker_threshold:
            # Clip gradient
            diff = frame_tensor - self.running_average
            diff = torch.clamp(diff, -self.gradient_clip, self.gradient_clip)
            frame_tensor = self.running_average + diff

            # Update running average
            self.running_average = self.alpha * frame_tensor + (1 - self.alpha) * self.running_average
            result = self.running_average
        else:
            # No flicker, use original frame and update running average
            self.running_average = frame_tensor.clone()
            result = frame_tensor

        self.last_frame = frame_tensor.clone()

        # Convert back to uint8 and scale to [0, 255]
        result = (result * 255.0).clamp(0, 255).to(torch.uint8)
        return result

    def reset(self):
        self.running_average = None
        self.last_frame = None

class SelectiveBrightSmoother:
    def __init__(self, brightness_threshold=0.8, smoothing_strength=0.3, device='cuda'):
        """
        Apply smoothing only to bright regions that are flickering

        Args:
            brightness_threshold: Threshold for considering pixels "bright"
            smoothing_strength: How much to smooth (0.0 = no smoothing, 1.0 = full smoothing)
            device: GPU device
        """
        self.brightness_threshold = brightness_threshold
        self.smoothing_strength = smoothing_strength
        self.device = device
        self.prev_frame = None

    def process_frame(self, frame_tensor):
        """
        Process frame with selective bright region smoothing
        """
        # Convert to float and scale to [0, 1]
        frame_tensor = frame_tensor.float() / 255.0

        if self.prev_frame is None:
            self.prev_frame = frame_tensor.clone()
            # Convert back to uint8 and scale to [0, 255]
            result = (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)
            return result

        # Calculate luminance (brightness)
        if frame_tensor.shape[0] == 3:  # RGB format (C, H, W)
            luminance_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device).view(3, 1, 1)
            luminance = torch.sum(frame_tensor * luminance_weights, dim=0, keepdim=True)
        else:
            luminance = torch.mean(frame_tensor, dim=0, keepdim=True)

        # Create mask for bright regions
        bright_mask = (luminance > self.brightness_threshold).float()

        # Calculate difference from previous frame
        diff = torch.abs(frame_tensor - self.prev_frame)
        flicker_mask = (torch.mean(diff, dim=0, keepdim=True) > 0.05).float()

        # Combine masks: only smooth bright AND flickering regions
        smooth_mask = bright_mask * flicker_mask

        # Apply smoothing only to selected regions
        smoothed = self.smoothing_strength * self.prev_frame + (1 - self.smoothing_strength) * frame_tensor
        result = frame_tensor * (1 - smooth_mask) + smoothed * smooth_mask

        self.prev_frame = frame_tensor.clone()

        # Convert back to uint8 and scale to [0, 255]
        result = (result * 255.0).clamp(0, 255).to(torch.uint8)
        return result

    def reset(self):
        self.prev_frame = None
class ExponentialMovingAverageSmoother:
    def __init__(self, alpha=0.7, flicker_threshold=0.1, device='cuda'):
        """
        Exponential moving average smoother - very fast, minimal memory

        Args:
            alpha: Smoothing factor (0.0 = all smoothing, 1.0 = no smoothing)
            flicker_threshold: Apply smoothing only when flicker is detected
            device: GPU device
        """
        self.alpha = alpha
        self.flicker_threshold = flicker_threshold
        self.device = device
        self.running_average = None
        self.last_frame = None

    def process_frame(self, frame_tensor):
        """
        Process frame with exponential moving average
        """
        # Convert to float and scale to [0, 1]
        frame_tensor = frame_tensor.float() / 255.0

        if self.running_average is None:
            self.running_average = frame_tensor.clone()
            self.last_frame = frame_tensor.clone()
            # Convert back to uint8 and scale to [0, 255]
            result = (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)
            return result

        # Calculate flicker amount
        if self.last_frame is not None:
            flicker = torch.mean(torch.abs(frame_tensor - self.last_frame)).item()
        else:
            flicker = 0.0
        print(flicker)
        # Apply smoothing only if flicker is detected
        if flicker > self.flicker_threshold:
            # Update running average
            self.running_average = self.alpha * frame_tensor + (1 - self.alpha) * self.running_average
            result = self.running_average
        else:
            # No flicker, use original frame and update running average
            self.running_average = frame_tensor.clone()
            result = frame_tensor

        self.last_frame = frame_tensor.clone()

        # Convert back to uint8 and scale to [0, 255]
        result = (result * 255.0).clamp(0, 255).to(torch.uint8)
        return result

    def reset(self):
        self.running_average = None
        self.last_frame = None

class FastOutlierResistantSmoother:
    def __init__(self, window_size=3, outlier_threshold=0.3, device='cuda'):
        """
        Fast outlier-resistant smoother for streaming frames

        Args:
            window_size: Number of frames to consider
            outlier_threshold: How much a pixel can deviate before being considered an outlier
            device: GPU device
        """
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        self.device = device
        self.frame_buffer = []

        # Pre-compute weights (center-heavy)
        if window_size == 3:
            self.weights = torch.tensor([0.2, 0.6, 0.2], device=device)
        elif window_size == 5:
            self.weights = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device)
        else:
            # Gaussian-like weights
            weights = torch.exp(-0.5 * ((torch.arange(window_size, device=device) - window_size//2) / (window_size/4))**2)
            self.weights = weights / weights.sum()

        self.weights = self.weights.view(-1, 1, 1, 1)

    def process_frame(self, frame_tensor):
        """
        Process frame with outlier detection and fast smoothing
        """
        # Convert to float and scale to [0, 1]
        frame_tensor = frame_tensor.float() / 255.0

        # Add to buffer
        self.frame_buffer.append(frame_tensor.clone())

        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) < self.window_size:
            # Convert back to uint8 and scale to [0, 255]
            result = (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8)
            return result

        # Stack frames
        stacked_frames = torch.stack(self.frame_buffer, dim=0)

        # Fast outlier detection: check if current frame deviates too much from mean
        mean_frame = torch.mean(stacked_frames[:-1], dim=0)  # Mean of previous frames
        current_frame = stacked_frames[-1]

        # Calculate absolute difference
        diff = torch.abs(current_frame - mean_frame)

        # Create outlier mask (1 where outlier, 0 where normal)
        outlier_mask = (diff > self.outlier_threshold).float()

        # For outlier pixels, use smoothed value; for normal pixels, use current frame
        smoothed = torch.sum(stacked_frames * self.weights, dim=0)
        result = current_frame * (1 - outlier_mask) + smoothed * outlier_mask

        # Convert back to uint8 and scale to [0, 255]
        result = (result * 255.0).clamp(0, 255).to(torch.uint8)
        return result

    def reset(self):
        self.frame_buffer = []

class StreamingMedianFilter:
    def __init__(self, window_size=3, device='cuda'):
        """
        Real-time median filter for a stream of PyTorch tensors on GPU

        Args:
            window_size: Number of frames to use for median calculation (must be odd)
            device: GPU device
        """
        assert window_size % 2 == 1, "Window size must be odd"
        self.window_size = window_size
        self.device = device
        self.frame_buffer = []

    def process_frame(self, frame_tensor):
        """
        Process a single frame from the stream

        Args:
            frame_tensor: PyTorch tensor on GPU (C, H, W) or (H, W, C)

        Returns:
            Median-filtered tensor on GPU
        """
        # Add to buffer
        if frame_tensor.dtype == torch.uint8:
            frame_tensor = frame_tensor.float() / 255.0
        self.frame_buffer.append(frame_tensor.clone())

        # Keep only window_size frames
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

        # If not enough frames, return original
        if len(self.frame_buffer) < self.window_size:
            return frame_tensor.to(torch.uint8)

        # Stack frames along a new dimension
        stacked_frames = torch.stack(self.frame_buffer, dim=0)  # (window_size, C, H, W) or (window_size, H, W, C)

        # Calculate median along the time dimension
        median_frame = torch.median(stacked_frames, dim=0).values

        return (median_frame*255).to(torch.uint8)

    def reset(self):
        """Clear the frame buffer"""
        self.frame_buffer = []

class AdaptiveBrightnessFlickerRemover:
    """More sophisticated version with adaptive smoothing based on brightness level"""

    def __init__(self, window_size=3, min_brightness=0.6, max_brightness=0.9, device='cuda'):
        """
        Args:
            window_size: Number of frames to consider for smoothing
            min_brightness: Brightness level where smoothing begins (0.0-1.0)
            max_brightness: Brightness level where smoothing is maximum (0.0-1.0)
            device: GPU device
        """
        self.window_size = window_size
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.device = device
        self.frame_buffer = []

        # Pre-compute weights
        if window_size == 3:
            self.weights = torch.tensor([0.25, 0.5, 0.25], device=device)
        elif window_size == 5:
            self.weights = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device)
        else:
            self.weights = torch.ones(window_size, device=device) / window_size

        self.weights = self.weights.view(-1, 1, 1, 1)

    def smooth_frame(self, frame_tensor):
        """
        Apply adaptive temporal smoothing based on brightness

        Args:
            frame_tensor: PyTorch tensor of shape (C, H, W) on GPU, values in [0,1]

        Returns:
            Smoothed frame tensor on GPU
        """
        # Add to buffer
        if frame_tensor.dtype == torch.uint8:
            frame_tensor = frame_tensor.float() / 255.0
        self.frame_buffer.append(frame_tensor.clone())

        # Keep only window_size frames
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

        # If not enough frames, return original
        if len(self.frame_buffer) < self.window_size:
            return frame_tensor.to(torch.uint8)


        # Stack frames
        stacked_frames = torch.stack(self.frame_buffer, dim=0)  # (window_size, C, H, W)

        # Calculate luminance
        if stacked_frames.shape[1] == 3:  # RGB
            luminance_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device).view(1, 3, 1, 1)
            luminance = torch.sum(stacked_frames * luminance_weights, dim=1, keepdim=True)
        else:
            luminance = torch.mean(stacked_frames, dim=1, keepdim=True)

        # Get current frame luminance
        current_luminance = luminance[-1]  # (1, H, W)

        # Create adaptive mask based on brightness
        # 0.0 for pixels below min_brightness, 1.0 for pixels above max_brightness,
        # and linearly interpolated in between
        blend_factor = ((current_luminance - self.min_brightness) /
                        (self.max_brightness - self.min_brightness)).clamp(0, 1)

        # Apply weighted average to stacked frames
        smoothed = torch.sum(stacked_frames * self.weights, dim=0)

        # Blend original and smoothed based on brightness-adaptive mask
        result = frame_tensor * (1 - blend_factor) + smoothed * blend_factor

        return (result*255).to(torch.uint8)

    def reset(self):
        self.frame_buffer = []

class BrightRegionFlickerRemover:
    def __init__(self, window_size=3, brightness_threshold=0.7, device='cuda'):
        """
        Specialized flicker remover that focuses on bright regions

        Args:
            window_size: Number of frames to consider for smoothing
            brightness_threshold: Threshold above which pixels are considered "bright" (0.0-1.0)
            device: GPU device
        """
        self.window_size = window_size
        self.brightness_threshold = brightness_threshold
        self.device = device
        self.frame_buffer = []

        # Pre-compute weights for temporal smoothing
        if window_size == 3:
            self.weights = torch.tensor([0.25, 0.5, 0.25], device=device)
        elif window_size == 5:
            self.weights = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device)
        else:
            self.weights = torch.ones(window_size, device=device) / window_size

        # Reshape weights for broadcasting
        self.weights = self.weights.view(-1, 1, 1, 1)

    def smooth_frame(self, frame_tensor):
        """
        Apply temporal smoothing only to bright regions of the frame

        Args:
            frame_tensor: PyTorch tensor of shape (C, H, W) on GPU, values in [0,1]

        Returns:
            Smoothed frame tensor on GPU
        """
        # Ensure frame is float for blending
        if frame_tensor.dtype == torch.uint8:
            frame_tensor = frame_tensor.float() / 255.0
        # Add to buffer
        self.frame_buffer.append(frame_tensor.clone())

        # Keep only window_size frames
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

        # If not enough frames, return original
        if len(self.frame_buffer) < self.window_size:
            return frame_tensor.to(torch.uint8)

        # Stack frames
        stacked_frames = torch.stack(self.frame_buffer, dim=0)  # (window_size, C, H, W)

        # Calculate luminance (brightness) - simple RGB average or use proper coefficients
        # For RGB: Y = 0.2126*R + 0.7152*G + 0.0722*B
        if stacked_frames.shape[1] == 3:  # RGB
            luminance_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device).view(1, 3, 1, 1)
            luminance = torch.sum(stacked_frames * luminance_weights, dim=1, keepdim=True)  # (window_size, 1, H, W)
        else:
            # Simple average for non-RGB data
            luminance = torch.mean(stacked_frames, dim=1, keepdim=True)  # (window_size, 1, H, W)

        # Get current frame luminance
        current_luminance = luminance[-1]  # (1, H, W)

        # Create bright region mask (1 for bright pixels, 0 for dark)
        bright_mask = (current_luminance > self.brightness_threshold).float()  # (1, H, W)

        # Apply weighted average to stacked frames
        smoothed = torch.sum(stacked_frames * self.weights, dim=0)  # (C, H, W)

        # Blend original and smoothed based on brightness mask
        # Only apply smoothing to bright regions
        result = frame_tensor * (1 - bright_mask) + smoothed * bright_mask

        return (result*255).to(torch.uint8)

    def reset(self):
        """Reset the frame buffer"""
        self.frame_buffer = []

class GPUTemporalSmoother:
    def __init__(self, window_size=3, device='cuda'):
        self.window_size = window_size
        self.device = device
        self.frame_buffer = []

        # Pre-compute weights on GPU
        if window_size == 3:
            self.weights = torch.tensor([0.25, 0.5, 0.25], device=device, dtype=torch.float32)
        elif window_size == 5:
            self.weights = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device, dtype=torch.float32)
        else:
            # Uniform weights
            weight = 1.0 / window_size
            self.weights = torch.full((window_size,), weight, device=device, dtype=torch.float32)

    def smooth_frame(self, frame_tensor):
        """
        Apply temporal smoothing to a single frame tensor on GPU.

        Args:
            frame_tensor: PyTorch tensor of shape (H, W, C) or (C, H, W) on GPU

        Returns:
            Smoothed frame tensor on GPU
        """
        # Ensure frame is float for blending
        if frame_tensor.dtype == torch.uint8:
            frame_tensor = frame_tensor.float() / 255.0

        # Add to buffer
        self.frame_buffer.append(frame_tensor.clone())

        # Keep only window_size frames
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

        # If not enough frames, return original
        if len(self.frame_buffer) < self.window_size:
            return frame_tensor

        # Stack frames and apply weights
        stacked_frames = torch.stack(self.frame_buffer, dim=0)  # (window_size, H, W, C) or (window_size, C, H, W)

        # Reshape weights to match tensor dimensions
        if len(stacked_frames.shape) == 4:  # (window_size, H, W, C)
            weights_reshaped = self.weights.view(-1, 1, 1, 1)
        else:  # (window_size, C, H, W)
            weights_reshaped = self.weights.view(-1, 1, 1, 1)

        # Weighted average
        smoothed = torch.sum(stacked_frames * weights_reshaped, dim=0)

        return (smoothed*255).to(torch.uint8)

    def reset(self):
        """Reset the frame buffer"""
        self.frame_buffer = []

def smooth_frame_sequence_gpu(frames, window_size=3):
    """
    Apply temporal smoothing to a sequence of frames on GPU.

    Args:
        frames: List of PyTorch tensors on GPU
        window_size: Size of temporal window

    Returns:
        List of smoothed frame tensors on GPU
    """
    smoother = GPUTemporalSmoother(window_size, device=frames[0].device)
    smoothed_frames = []

    for frame in frames:
        smoothed_frame = smoother.smooth_frame(frame)
        smoothed_frames.append(smoothed_frame)

    return smoothed_frames

def gaussian_temporal_smoothing_torch(frames, kernel_size=3, sigma=1.0):
    """
    Applies Gaussian smoothing along temporal dimension using PyTorch.

    Args:
        frames (torch.Tensor): Tensor of shape (T, C, H, W)
        kernel_size (int): Size of Gaussian kernel (must be odd)
        sigma (float): Standard deviation of Gaussian kernel

    Returns:
        torch.Tensor: Smoothed frames
    """
    device = frames.device
    T, C, H, W = frames.shape

    # Create 1D Gaussian kernel
    x = torch.arange(-kernel_size // 2, kernel_size // 2 + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # Pad frames for temporal convolution
    pad_size = kernel_size // 2
    frames_padded = F.pad(frames, (0, 0, 0, 0, 0, 0, pad_size, pad_size), mode='replicate')

    # Reshape for 1D convolution along time dimension
    frames_reshaped = frames_padded.permute(1, 2, 3, 0).reshape(C * H * W, 1, T + 2 * pad_size)

    # Apply 1D convolution
    kernel_reshaped = kernel.view(1, 1, kernel_size)
    smoothed_reshaped = F.conv1d(frames_reshaped, kernel_reshaped, padding=0)

    # Reshape back to original format
    smoothed = smoothed_reshaped.reshape(C, H, W, T).permute(3, 0, 1, 2)

    return smoothed

class MultiScaleStreamingEMA:
    def __init__(self, alphas=[0.8, 0.6, 0.5], device='cuda'):
        self.filters = [StreamingEMA(alpha, device) for alpha in alphas]
        self.device = device

    def process_frame(self, current_frame):
        # Apply multiple smoothing levels
        results = []
        for filter in self.filters:
            smoothed = filter.process_frame(current_frame)
            results.append(smoothed)

        # Combine results (weighted average)
        weights = torch.tensor([0.5, 0.3, 0.2], device=self.device)
        combined = sum(w * frame for w, frame in zip(weights, results))

        return combined.to(torch.uint8)
class StreamingEMA:
    def __init__(self, alpha=0.7, device='cuda'):
        self.alpha = alpha
        self.device = device
        self.previous_frame = None

    def process_frame(self, current_frame):
        """
        Process a single frame in the stream
        current_frame: tensor of shape (C, H, W) or (H, W, C)
        """
        current_frame = current_frame.to(self.device)

        if self.previous_frame is None:
            # First frame - no smoothing
            self.previous_frame = current_frame.clone()
            return current_frame.to(torch.uint8)

        # Apply exponential moving average
        smoothed_frame = (self.alpha * current_frame +
                         (1 - self.alpha) * self.previous_frame)

        # Update state for next frame
        self.previous_frame = smoothed_frame.clone().to(torch.uint8)

        return smoothed_frame.to(torch.uint8)

    def reset(self):
        """Reset the filter state"""
        self.previous_frame = None

from io import BytesIO
import time
from pydub import AudioSegment

class AudioPacketizer:
    def __init__(self, mp3_path: str, packet_duration_ms: int = 50, force_mono: bool = False, format: str = 'opus'):
        """
        Initialize the audio packetizer
        Args:
            mp3_path: Path to MP3 file
            packet_duration_ms: Duration of each packet in milliseconds (default: 50ms)
            force_mono: If True, converts stereo to mono. If False, maintains original channels (default: False)
            format: Audio format to use ('opus' or 'aac')
        """
        self.audio = AudioSegment.from_mp3(mp3_path)
        self.packet_duration_ms = packet_duration_ms
        self.position = 0
        self.overlap_ms = 15
        self.format = format.lower()
        self.exit = False

        # Convert to mono if requested
        if force_mono and self.audio.channels > 1:
            self.audio = self.audio.set_channels(1)

        self.channels = self.audio.channels

        # Set sample rate based on format
        if self.format == 'opus':
            self.audio = self.audio.set_frame_rate(48000)  # Opus works best with 48kHz
        elif self.format == 'aac': # AAC
            self.audio = self.audio.set_frame_rate(44100)  # AAC standard
        else:
            raise(f"{self.format} audio format not supported")

        # Set consistent format
        self.audio = self.audio.set_sample_width(2)  # 16-bit

    def get_next_packet(self) -> tuple[bytes, int] | None:
        """
        Get the next audio packet
        Returns:
            Tuple of (packet_data, timestamp) or None if end of audio
        """
        if self.position >= len(self.audio) or self.exit:
            return None

        # Extract packet
        end_pos = min(self.position + self.packet_duration_ms, len(self.audio))
        packet = self.audio[self.position:end_pos]
        packet = packet.fade_in(self.overlap_ms).fade_out(self.overlap_ms)

        # Export packet
        buffer = BytesIO()

        if self.format == 'opus':
            # Opus-specific export parameters
            export_params = [
                "-ar", "48000",          # Sample rate (48kHz optimal for Opus)
                "-ac", str(self.channels),  # Number of channels
                "-acodec", "libopus",    # Use Opus codec
                "-b:a", "64k",           # Bitrate
                "-vbr", "on",            # Variable bitrate
                "-compression_level", "10",  # Compression level
                "-frame_duration", "20",  # Frame size in ms
                "-application", "audio"   # Application type
            ]

            # Add channel-specific settings for Opus
            if self.channels == 1:
                export_params.extend(["-cutoff", "12000"])
            else:
                export_params.extend(["-cutoff", "20000"])

            packet.export(buffer, format='opus', parameters=export_params)
        else:
            # AAC-specific export parameters
            export_params = [
                "-ar", "44100",      # Sample rate
                "-ac", str(self.channels),  # Number of channels
                "-acodec", "aac",    # Use AAC codec
                "-b:a", "192k",      # Higher bitrate for AAC
                "-q:a", "2",         # Quality setting
                "-cutoff", "18000"   # Frequency cutoff
            ]

            packet.export(buffer, format='adts', parameters=export_params)

        packet_data = buffer.getvalue()

        # Calculate timestamp
        timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds

        # Update position with overlap
        self.position = end_pos - self.overlap_ms

        return packet_data, timestamp
    def reset(self):
    #         """Reset the position to the beginning of the audio"""
        self.position = 0

# class AudioPacketizer:
#     def __init__(self, mp3_path: str, packet_duration_ms: int = 50, force_mono: bool = False):
#         """
#         Initialize the audio packetizer
#         Args:
#             mp3_path: Path to MP3 file
#             packet_duration_ms: Duration of each packet in milliseconds (default: 50ms)
#             force_mono: If True, converts stereo to mono. If False, maintains original channels (default: False)
#         """
#         self.audio = AudioSegment.from_mp3(mp3_path)
#         self.packet_duration_ms = packet_duration_ms
#         self.position = 0
#         self.overlap_ms = 15

#         # Convert to mono if requested
#         if force_mono and self.audio.channels > 1:
#             self.audio = self.audio.set_channels(1)

#         self.channels = self.audio.channels

#         # Ensure consistent sample rate (44.1kHz is standard)
#         self.audio = self.audio.set_frame_rate(44100)

#         # Set consistent format
#         self.audio = self.audio.set_sample_width(2)  # 16-bit

#     def get_next_packet(self) -> tuple[bytes, int] | None:
#         """
#         Get the next audio packet
#         Returns:
#             Tuple of (packet_data, timestamp) or None if end of audio
#         """
#         if self.position >= len(self.audio):
#             return None

#         # Extract packet
#         end_pos = min(self.position + self.packet_duration_ms, len(self.audio))
#         packet = self.audio[self.position:end_pos]
#         packet = packet.fade_in(self.overlap_ms).fade_out(self.overlap_ms)

#         # Export packet as MP3
#         buffer = BytesIO()

#         # Adjust export parameters based on number of channels
#         export_params = [
#             "-ar", "44100",      # Sample rate
#             "-ac", str(self.channels),  # Number of channels (1 or 2)
#             "-b:a", "192k",      # Higher bitrate for Windows
#             "-q:a", "0",         # Highest quality
#             "-bufsize", "192k",  # Larger buffer size
#             "-compression_level", "0"  # Fastest compression
#         ]

#         # Add joint stereo parameter only for stereo audio
#         if self.channels == 2:
#             export_params.extend(["-joint_stereo", "0"])  # Disable joint stereo for stereo

#         packet.export(buffer,
#                      format='mp3',
#                      parameters=export_params)

#         packet_data = buffer.getvalue()

#         # Calculate timestamp
#         timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds

#         # Update position with overlap
#         self.position = end_pos - self.overlap_ms

#         return packet_data, timestamp

# class AudioPacketizer:
#     def __init__(self, mp3_path: str, packet_duration_ms: int = 50):
#         """
#         Initialize the audio packetizer
#         Args:
#             mp3_path: Path to MP3 file
#             packet_duration_ms: Duration of each packet in milliseconds (default: 22ms)
#         """
#         self.audio = AudioSegment.from_mp3(mp3_path)
#         self.packet_duration_ms = packet_duration_ms
#         self.position = 0
#         self.overlap_ms = 15

#         # Convert to mono if stereo
#         if self.audio.channels > 1:
#             self.audio = self.audio.set_channels(1)
#         assert self.audio.channels == 1, "Audio must be mono"  
#            # Ensure consistent sample rate (44.1kHz is standard)  
#         self.audio = self.audio.set_frame_rate(44100)  

#         # Set consistent format  
#         self.audio = self.audio.set_sample_width(2)  # 16-bit


#     def get_next_packet(self) -> tuple[bytes, int] | None:
#         """
#         Get the next audio packet
#         Returns:
#             Tuple of (packet_data, timestamp) or None if end of audio
#         """
#         if self.position >= len(self.audio):
#             return None

#         # Extract packet
#         end_pos = min(self.position + self.packet_duration_ms, len(self.audio))
#         packet = self.audio[self.position:end_pos]
#         packet = packet.fade_in(self.overlap_ms).fade_out(self.overlap_ms)  

#         # Export packet as MP3
#         buffer = BytesIO()
#         packet.export(buffer, format='mp3', 
#                       parameters=[
#                         "-ar", "44100",  # Sample rate
#                         "-ac", "1",      # Mono
#                         "-b:a", "192k",     # Higher bitrate for Windows  
#                         "-joint_stereo", "0",  # Disable joint stereo
#                         "-q:a", "0",     # Highest quality
#                         "-bufsize", "192k", # Larger buffer size
#                         "-compression_level", "0"  # Fastest compression
#                         ]
#                       )
#         # packet.export(buffer,
#         #             format='ipod',
#         #             parameters=[
#         #             "-ar", "44100",
#         #             "-ac", "1",
#         #             "-b:a", "96k",
#         #             "-q:a", "0"
#         #         ]
               
                   
#         # )
#         packet_data = buffer.getvalue()

#         # Calculate timestamp
#         timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds

#         # Update position
#         # self.position = end_pos
#         self.position = end_pos - self.overlap_ms  


#         return packet_data, timestamp

#     def reset(self):
#         """Reset the position to the beginning of the audio"""
#         self.position = 0


  
        
class Viewer():

    def __init__(self, seq, exp,  port, title="",f_ratio=0.8, w=1920, h=1080, near=0.01, far=100.0):
        self.seq = seq
        self.exp = exp
        self.viser_server = viser.ViserServer(port=port, title=title)
        self.viser_server.scene.world_axes.visible = False
        self.clients_num = 0
        self.k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.scene_data, _ , self.look_at = self._load_scene_data4(self.seq, self.exp, seg_as_col=False)
        self.render_viewers: Dict[int, RenderViewers] = {}
        signal.signal(signal.SIGINT, self.signal_handler)
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_disconnect(self.handle_disconnect_client)
        self.running = True
   
        



    def handle_disconnect_client(self, client:viser.ClientHandle):
        print(f"{client.client_id} client disconnected!")
        self.render_viewers[client.client_id].running = False
        self.render_viewers[client.client_id].thread_cuda.join()
        self.render_viewers[client.client_id].thread_encode.join()
        self.render_viewers[client.client_id].thread_process_video_buffers.join()
        self.render_viewers[client.client_id].thread_process_audio_buffers.join()

        self.render_viewers.pop(client.client_id)



    
    def handle_new_client(self, client:viser.ClientHandle):
        
            
        self.clients_num +=1 
        # Show the client ID in the GUI.
        # gui_info = client.gui.add_text("Client ID", initial_value= str(client.client_id))
        
        # gui_info.disabled = False
        # button = client.gui.add_button("Start/Pause Sound")
        # button.disabled = False
        print("new client!", client.client_id)
        print("Total number of clients connected to Hamit's demo:", len(self.render_viewers))
        self.render_viewers[client.client_id] = RenderViewers(self, client)
        
        self.render_viewers[client.client_id].start()

   
   

    def _load_scene_data(self, params, low_upper_limit, seg_as_col=False):
        
        
        is_fg = params['seg_colors'][:, 0] > 0.5
        scene_data = []
        for t in range(*low_upper_limit):
            
            rendervar = {
                'means3D': params['means3D'][t].cuda(),
                'colors_precomp': params['rgb_colors'][t].cuda() if not seg_as_col else params['seg_colors'].cuda(),
                'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t].cuda()),
                'opacities': torch.sigmoid(params['logit_opacities']).cuda(),
                'scales': torch.exp(params['log_scales']).cuda(),
                'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
            }
            # rendervar = {k: v.cuda() for k, v in rendervar.items()}
            if REMOVE_BACKGROUND:
                rendervar = {k: v[is_fg] for k, v in rendervar.items()}
            scene_data.append(rendervar)
        if REMOVE_BACKGROUND:
            is_fg = is_fg[is_fg]
        return scene_data, is_fg
    
    def _load_scene_data2(self, seq, exp, seg_as_col=False):
        
        params = dict(np.load(f"./output/{exp}/{seq}/params_9.npz"))
    

        params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
        is_fg = params['seg_colors'][:, 0] > 0.5
        scene_data = []
        length = len(params['means3D'])
        for t in range(length): #len(params['means3D'])):
            rendervar = {
                'means3D': params['means3D'][t],
                'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
                'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
                'opacities': torch.sigmoid(params['logit_opacities']),
                'scales': torch.exp(params['log_scales']),
                'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
            }
            if REMOVE_BACKGROUND:
                rendervar = {k: v[is_fg] for k, v in rendervar.items()}
            scene_data.append(rendervar)
        if REMOVE_BACKGROUND:
            is_fg = is_fg[is_fg]
        return scene_data, is_fg
    def _load_scene_data3(self, seq, exp, seg_as_col=False):
        params_file =[ os.path.join(f"./output/{exp}/{seq}/", file) for file in  os.listdir(f"./output/{exp}/{seq}/") if file.startswith("params")]
        params_file = sorted(params_file, key= lambda x : int(os.path.basename(x).split("_")[1].split(".")[0]) if len(os.path.basename(x).split("_")) > 1 else os.path.basename(x).split("_")[0].split(".")[0]) 
        pc = np.load(os.path.join(f"./output/{exp}/{seq}/", "init_pt_cld.npz"))

        xyz = [vert[:3] for vert in pc['data']]
        xyz = np.asarray(xyz)
        center = np.mean(xyz[:], axis=0)
        print("Foreground center:",center)
        # print(params_file)
        scene_data = []
        total = 0
        total_cnt =0
        for l, param_file  in enumerate(params_file):
            # if l > 0:
            #     break
           
            params = dict(np.load(param_file))  
            print(f"{param_file} loaded!")

            params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
            is_fg = params['seg_colors'][:, 0] > 0.5
            # if l == 69:
            #     length=80
            # elif l ==75:
            #     length = 5
            # else:
            length = len(params['means3D'])
            total = total + length
            print(f"total timesteps:", total, l)
            for t in range(length): #len(params['means3D'])):
                if total_cnt > 15:
                    break
                rendervar = {
                    'means3D': params['means3D'][t],
                    'colors_precomp':  params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
                    # 'colors_precomp':  torch.sigmoid(params['rgb_colors'][t]) if not seg_as_col else params['seg_colors'],
                    'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
                    'opacities': torch.sigmoid(params['logit_opacities']),
                    'scales': torch.exp(params['log_scales']),
                    'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
                }
                if REMOVE_BACKGROUND:
                    rendervar = {k: v[is_fg] for k, v in rendervar.items()}
                scene_data.append(rendervar)
                total_cnt += 1

            if total_cnt > 57:
                break
            if REMOVE_BACKGROUND:
                is_fg = is_fg[is_fg]
        return scene_data, is_fg, center
    def _load_scene_data4(self, seq, exp, seg_as_col=False):
        params_files =[ os.path.join(f"./output/{exp}/{seq}/", file) for file in  os.listdir(f"./output/{exp}/{seq}/") if file.startswith("params")]
        params_files = sorted(params_files, key= lambda x : int(os.path.basename(x).split("_")[1].split(".")[0]) if len(os.path.basename(x).split("_")) > 1 else os.path.basename(x).split("_")[0].split(".")[0]) 
        pc = np.load(os.path.join(f"./output/{exp}/{seq}/", "init_pt_cld.npz"))

        xyz = [vert[:3] for vert in pc['data']]
        xyz = np.asarray(xyz)
        center = np.mean(xyz[:], axis=0)
        print("Foreground center:",center)
        # print(params_file)
        scene_data = []
        total_length = 0
        tot_cnt = 0

        for l, params_file  in enumerate(params_files):
          
            params = np.load(params_file, allow_pickle=True)
            print(f"{params_file} loaded!")
  

            for param in params:
                if param == "allow_pickle":
                    continue
                data = params[param]
                
                for pr in data:
                    total_length += len(pr)
                    for id, p in enumerate(pr):
                        # if tot_cnt < 42:
                        #     tot_cnt += 1
                        #     continue
                        if tot_cnt > 40:
                            break
                        rendervar = {
                            'means3D': torch.tensor(p['means3D']).cuda().float(),
                            'colors_precomp':  torch.tensor(p['rgb_colors']).cuda().float() if not seg_as_col else torch.tensor(p['seg_colors']).cuda().float(),
                            'rotations': torch.nn.functional.normalize( torch.tensor(p['unnorm_rotations']).cuda().float()),
                            'opacities': torch.sigmoid(torch.tensor(pr[0]['logit_opacities']).cuda().float()),
                            'scales': torch.exp(torch.tensor(pr[0]['log_scales']).cuda().float()),
                            'means2D': torch.zeros_like( torch.tensor(pr[0]['means3D']).float(), device="cuda")
                        }

                        scene_data.append(rendervar)
                        tot_cnt += 1


        is_fg = False #params['seg_colors'][:, 0] > 0.5
        print("total length:", total_length) 
        
        return scene_data, is_fg, center

    def start_viewer(self):
    
        while True:
            if not self.running:
                # time.sleep(1)
                break
            print("Total number of clients connected to the demo:", len(self.render_viewers))
            time.sleep(3600)
    def signal_handler(self,sig, frame):

        print('You pressed Ctrl+C!')

        self.running = False
        for key, val in self.render_viewers.items():
            # val.video_audio_event.set()
            val.packetizer.exit = True
            val.running = self.running
            val.thread_cuda.join()
            val.thread_encode.join()
            val.thread_process_video_buffers.join()
            val.thread_process_audio_buffers.join()

            
        
        viewer.viser_server.stop()

        sys.exit(0)
    
   

class RenderViewers():
  

    # # look_at = np.array([ 0.08500356,  0.42318333, -0.73812744 ]) #np.array([-0.28, 1.65, 0.09]) 
    # roll_limit = (np.pi, -np.pi) 
    # # roll_limit = (1.4, -1.0)
    # pitch_limit = (2.4, -1.3)
  
    distance_in = 1  
    distance_out = 25

    theta_limits = (60, 100)
    # phi_limits = (-160, -10)
    phi_limits = (-360, 360)

  

  
    

    def __init__(self, viewer, client ):
        self.viewer = viewer
        self.client = client
        self.encode_event = threading.Event()
        self.cuda_event = threading.Event()
        self.audio_event = threading.Event()

        self.thread_cuda = threading.Thread(target=self.render_images)    
        self.thread_encode = threading.Thread(target=self.encode_image)
        self.thread_process_audio_buffers = threading.Thread(target=self.process_audio_buffers)

        self.thread_process_video_buffers =  threading.Thread(target=self.process_video_buffers)
        self.running = True
        self.triggered = False
        self.scene_data = viewer.scene_data
        self.look_at = viewer.look_at
        # self.look_at = np.array([-0.879248,0.513477,2.993033])
   
        self.frame_rate = 30
        self.interval = 1.0/self.frame_rate
        length=len(self.scene_data) * self.interval
        # self.gui_progress_bar = client.gui.add_progress_bar(0, color="#228be6")
        self.gui_slider_bar_text = client.gui.add_slider_bar_text(0, length=length, color="#228be6")
       
        self.gui_play_button  = client.gui.add_button("Start_player", visible=True, icon=viser.Icon.PLAYER_PLAY, color="white", label_second="Pause_player",
                                                      icon_second=viser.Icon.PLAYER_PAUSE, color_second="white", visible_second=False)
        self.gui_player_skip_back_button= client.gui.add_button("Player_skip_back", visible=True,icon=viser.Icon.PLAYER_SKIP_BACK,  
                                                                color="white")
        self.gui_player_sound_button= client.gui.add_button("Player_sound_off", visible=True,icon=viser.Icon.VOLUME_OFF, color="white",
                                                            label_second="Player_sound_on",icon_second=viser.Icon.VOLUME, color_second="white",
                                                            visible_second=False)
       
        
        self.gui_play_button.on_click(self.handle_on_play_click)
        self.gui_player_skip_back_button.on_click(self.handle_on_play_skip_back_click)
        self.gui_player_sound_button.on_click(self.handle_on_play_sound_click)

        self.w = viewer.w
        self.h = viewer.h
        self.far = viewer.far   
        self.near = viewer.near
        self.k = viewer.k
        self.first_enter = False
        self.frame_number=0
        self.data_ready = False
     
        self.isPaused = True
        self.skip_back = False
        # self.increment=1
        self.mutex = threading.Lock()  

    
    def handle_on_play_click(self, _):
        self.gui_play_button.visible= not self.gui_play_button.visible
        self.gui_play_button.visible_second= not self.gui_play_button.visible_second

        
        self.isPaused = not self.isPaused 


    
    def handle_on_play_skip_back_click(self,_):
        self.gui_play_button.visible=True
        self.gui_play_button.visible_second=False
        # self.isPaused = True
        # self.gui_play_button.value=False
        # self.current_ts = len(self.scene_data)
        self.isPaused = True
        # time.sleep(0.1)

        self.skip_back = True
        self.gui_slider_bar_text.value=0.0
        # self.gui_text.value=f"{0:02d}:{0:02d}"

    def handle_on_play_sound_click(self,_):
        self.gui_player_sound_button.visible= not self.gui_player_sound_button.visible
        self.gui_player_sound_button.visible_second= not self.gui_player_sound_button.visible_second



    def handle_on_play_skip_forward_click(self,_):
        with self.mutex:  
            self.increment=20




        
    def start(self):
       

        self.thread_encode.start()
        time.sleep(0.5)

        self.thread_cuda.start()
       
        self.thread_process_video_buffers.start()
        # time.sleep(1)
        self.thread_process_audio_buffers.start()

   

    def process_audio_buffers(self, audio_file="/home/hamit/Downloads/ocean.mp3"):
    # This is an example of how to generate audio packets
    # Replace this with your actual audio packet generation logic
        frame_number = 0
        self.packetizer = AudioPacketizer(audio_file, packet_duration_ms= self.interval * 1000*5 , force_mono=False, format="opus") 
        interval = self.packetizer.packet_duration_ms / 1000.0  # Convert to seconds  


        while self.running :

            # t0 = time.time() 
            # packet  = self.packetizer.get_next_packet() 
            self.audio_event.wait()
            self.audio_event.clear()
            if not self.isPaused:
                packet  = self.packetizer.get_next_packet() 
            else:
                # time.sleep(self.interval)
                continue
            if packet is None:  
                self.packetizer.reset()  
                continue  
            packet_data, time_stamp = packet
            # audio_packets = self.generate_audio_packets(audio_file)
            # for pack in audio_packets:
            # print(packet_data)
            try:
                self.client.scene.set_background_audio_pckt (
                    packet_data,
                    frame_number
                )
            except Exception as e:
                print(f"Error sending audio packet: {e}")

            # More precise timing control
            # elapsed = time.time() - t0
            # sleep_time = interval - elapsed
            # if sleep_time > 0:
            #     time.sleep(sleep_time)
            
            

            frame_number += 1

        # Split audio into chunks
           
    
    def process_audio_buffers2(self, audio_file="/home/hamit/Downloads/yoga1.wav", segment_duration_ms=200):
    # def process_audio_buffers(self, audio_file="/home/hamit/Softwares/Dynamic3DGaussians/output/hamit_2024-12-04_17-14-42_scl_4_it_600_test1/2024-12-04_17-14-42/output.aac", segment_duration_ms=33):
        """
        Extracts audio segments corresponding to image timestamps.
        
        :param audio_file: Path to the audio file.
      
        :param segment_duration_ms: Duration of each audio segment in milliseconds.
        
        """

        # Load the audio file
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1)
        # audio.export("/home/hamit/Downloads/sample4.wav", format="wav")
        duration_ms = int(len(audio))
        print("duration in ms:", duration_ms)
        image_timestamps = [i for i in range(segment_duration_ms, duration_ms, segment_duration_ms)]  # Example timestamps in milliseconds
        
        frame_number = 0
       
        interval = segment_duration_ms/1000.0


        while self.running :

                # t0 = time.time() 
                start_time = 0
                for i, timestamp in enumerate(image_timestamps):
                    t0 = time.time() 

                    if not self.running:
                        break
                    # self.video_audio_event.wait()

                    start = time.time()
                    start_time = timestamp
                    end_time = start_time + segment_duration_ms
                    audio_segment = audio[start_time:end_time]
                    try:
                        self.client.scene.set_background_audio_pckt (
                                audio_segment.raw_data,
                                frame_number
                    )
                
            
                    except Exception as e:
                        print(f"Error sending audio packet: {e}")
                    
                    elapsed = time.time() - t0
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                   

                    end = time.time()

                    # print("Audio frame number", frame_number)
                    if frame_number % 50 ==0:      
                        print(f"{self.client.client_id} Audio Process Buffer fps:", 1.0/(end-start))  
                    frame_number += 1
                  

       
    
    
    def process_video_buffers(self):
        frame_number=0

        try:
            while self.running:
               
                start = time.time()
                size, data_ptr = gst_h264_endoder_pipeline.get_next_video_buffer_data(self.client_cnt)
                
                self.h264_pck = ctypes.string_at(data_ptr, size) 
                

                self.client.scene.set_background_h264_pckt(
                    self.h264_pck,
                    frame_number%300
                )
                
                end = time.time()

                if frame_number % 200 ==0:
                    print(f"{self.client.client_id} Sending encoded packet  fps:", 1/(end-start))  
                frame_number += 1
                 
        except KeyboardInterrupt:
            print("Stopped processing buffers.")
    
    def encode_image(self):
        self.client_cnt = gst_h264_endoder_pipeline.main_fun()
        print("Client num:",self.client_cnt)
        while self.running:
           
            self.encode_event.wait()
            self.encode_event.clear()
            
            if len(self.img) > 0:
                size = self.img.numel() * self.img.element_size()
                
                try:
                    
                    gst_h264_endoder_pipeline.push_tensor_frame(self.img.data_ptr(), size, self.frame_number, self.frame_rate, self.client_cnt)
                    
                    self.frame_number += 1


                except Exception as e:
                    print(f"Error encoding {self.img_num}: {e}")
              
            self.cuda_event.set()
        self.cuda_event.set()    

        gst_h264_endoder_pipeline.close_pipeline(self.client_cnt)
      
            
    def render_images(self):
        # ema_filter = StreamingEMA(alpha=0.7, device='cuda')  
        # ema_filter = MultiScaleStreamingEMA()
        # smoother = GPUTemporalSmoother(window_size=3, device='cuda')
        # smoother = FastOutlierResistantSmoother(window_size=5,outlier_threshold=0.8, device='cuda')
        # smoother = ExponentialMovingAverageSmoother(alpha=0.9, flicker_threshold=0.001, device='cuda')
        # smoother = SelectiveBrightSmoother(brightness_threshold=0.8, smoothing_strength=0.8, device='cuda')
        # smoother  = GradientClippedExponentialMovingAverageSmoother(alpha=0.7, flicker_threshold=0.05, gradient_clip=0.1, device='cuda')
        # smoother = YUVExponentialMovingAverageSmoother(alpha_y=0.8, alpha_uv=0.5, flicker_threshold=0.001, device='cuda')
        # smoother = ImprovedExponentialMovingAverageSmoother(alpha=0.9, flicker_threshold=0.001, adaptive_alpha=True, device='cuda')
        num_timestamps = len(self.scene_data)
        # quantizer = LuminanceQuantizer(luma_bits=6, device='cuda')
        stabilizer = SimpleGrayStabilizer(device='cuda', blend_factor=0.4)  

        # t0 = time.time() + self.interval
        c2w = np.eye(4)
        w2c = np.eye(4)

        
        while self.running:
            self.frame_number = 0 
            current_ts = 0
            previous_ts = 0
            # print("num_timestamps", num_timestamps)
            t = 0
            while t < num_timestamps:
            # for t in range(num_timestamps): 
            
                
                if not self.running:
                    break
                # print("current ts:", current_ts)
                old_proggress_bar_value = float((t / num_timestamps)*100)
                
                
                # self.gui_progress_bar.value = old_proggress_bar_value
                # sec_left = (num_timestamps - current_ts) * self.interval

                # minutes = int(sec_left // 60) 
                # seconds = int(sec_left) % 60
                # self.gui_text.value=f"{minutes:02d}:{seconds:02d}"
                # increment = self.increment )
                # increment = self.increment 
                while True:
                    t0 = time.time()
                    if not self.running:
                        break
                    if not self.first_enter: 
                        self.client.camera.wxyz = self.init_camera()[0]
                        self.client.camera.position = self.init_camera()[1] 
                        self.client.camera.look_at= self.look_at # for yoga 
                        self.first_enter = True

                    R_S03 = tf.SO3(np.asarray(self.client.camera.wxyz))
                    R = R_S03.as_matrix()
                    T = self.client.camera.position

                    vec_to_look_at = self.look_at - T
                    theta, phi = self.get_theta_phi_angles_from_cam_pos(vec_to_look_at)
                   
                        # (theta > 40 and theta < 110)  and  \
                    # if phi >= 0:
                    #     phi = -phi
                    # else:
                    #     phi = -phi
                        
                    # print("Theta and Phi in degrees:", theta, " ",  phi)
                    # if (np.linalg.norm(vec_to_look_at)  > self.distance_in) and \
                    #     (np.linalg.norm(vec_to_look_at) < self.distance_out):
                   
                    if (theta > self.theta_limits[0] and   theta < self.theta_limits[1] )  and (phi > self.phi_limits[0] and \
                        phi < self.phi_limits[1])  and  np.linalg.norm(vec_to_look_at)  > self.distance_in and \
                        np.linalg.norm(vec_to_look_at) < self.distance_out:

                        c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
                        w2c = np.linalg.inv(c2w)
                        # print(repr(c2w))
                    else:
                                
                        c2w = np.linalg.inv(w2c)

                        self.client.camera.position = c2w[:3,3] 
                        self.client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
                        self.client.camera.look_at = self.look_at 
                    # print(f"current ts:{current_ts}")
                    # self.img = self._render(w2c, self.scene_data[t], bg=[0, 177.0/255, 64.0/255])
                    img = self._render(w2c, self.scene_data[t], bg=[240./255, 240./255, 240./255])
                    self.img = reduce_color_depth_with_dithering(img, bits=6)
                    # self.img = reduce_color_depth_with_dithering(img, bits=6)
                    # torchvision.utils.save_image(self.img, "im.png")
                    # self.img = smoother.process_frame(img)  
                    # startTime= time.time()
                    # self.img = ema_filter.process_frame(img)  
                    # print(time.time()- startTime)
                    # if old_proggress_bar_value > 0:
                    
                    if old_proggress_bar_value + 50 < self.gui_slider_bar_text.value  or old_proggress_bar_value - 50 > self.gui_slider_bar_text.value:
                        # print(old_proggress_bar_value, self.gui_progress_bar.value)
                        if old_proggress_bar_value == 0 and not self.isPaused:
                            t = 0
                        else:
                            t = int(self.gui_slider_bar_text.value * num_timestamps/100)
                        self.gui_slider_bar_text.value = float((t / num_timestamps)*100)
                        old_proggress_bar_value = self.gui_slider_bar_text.value
                        if self.isPaused:
                            previous_ts = t  

                    else:
                        
                        self.gui_slider_bar_text.value = old_proggress_bar_value
                        
                       
                    self.encode_event.set()
                    if t % 5 == 0 :
                        self.audio_event.set()

                    self.cuda_event.wait()
                   
                    
                    # print("time step:",t)
                    elapsed = time.time() - t0
                    sleep_time = self.interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                   

                    self.cuda_event.clear()

                   
                       

                    if self.skip_back:
                        # print("it is breaking")
                        break

                    if not self.isPaused:
                        previous_ts =  t 
                        self.frame_number += 1
                        break
                    else:
                        t = previous_ts
                        # self.gui_progress_bar.value = float((t / num_timestamps)*100)

                
              

                if self.skip_back:
                    self.skip_back = False
                    # print("it is breaking")
                    break
            
                t += 1

        self.encode_event.set()
        self.audio_event.set()

   
                



    def _render(self, w2c, timestep_data, bg=[0,0,0]):
        with torch.no_grad():
            cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
            im, radi, _, = Renderer(raster_settings=cam)(**timestep_data)
            # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
            # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")
            # im = torch.flip(im, dims=[0])
            im =  im.permute(1,2,0).contiguous()
            im = (im.clamp(0,1)*255).to(torch.uint8)
            return im
    
    
    @classmethod
    def init_camera(cls, y_angle=180., center_dist=5., cam_height= 1.5, f_ratio=0.82):
        ry = y_angle * np.pi / 180
        w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -0.0],
                        [0.,         1., 0.,          cam_height],
                        [np.sin(ry), 0., np.cos(ry),  center_dist],
                        [0.,         0., 0.,          1.]])
        c2w = np.array([[-8.97115993e-01,  3.44142464e-02, -4.40452671e-01,
         3.17136791e+00],
       [-3.63098614e-04,  9.96903688e-01,  7.86314468e-02,
         1.39948448e+00],
       [ 4.41794934e-01,  7.07014562e-02, -8.94325746e-01,
         5.62253613e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
        # c2w = np.linalg.inv(w2c)
        wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
        return wxyz, c2w[:3,3]
    
    @classmethod
    def get_theta_phi_angles_from_cam_pos(cls, position):

        # Calculate the length (magnitude) of the position vector
        position_length = np.linalg.norm(position)

        # Calculate the polar angle (theta) relative to the y-axis
        # Theta = arccos(y / ||position||)
        theta = np.arccos(position[1] / position_length)
        phi = np.arctan2(position[2], position[0])

        # Convert theta to degrees
        theta_degrees = np.degrees(theta)
        phi_degrees = np.degrees(phi)  


        return theta_degrees, phi_degrees


if __name__ == "__main__":




    # exp_name = "2025-05-20_17-44-29_tahir_1_fullsize_scl_2_ai_enhanced_2"
    # sequence = "2025-05-20_17-44-29_tahir_1"

    # exp_name = "2025-05-20_18-21-50_tahir2_fullsize_scl_2_ai_contrast_alpha14_aligned"
    # sequence = "2025-05-20_18-21-50_tahir2_aligned"

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_fullsize_scl_2_contrast_alpha12_aligned"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1"

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_contrast_alpha12_aligned"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_scl_2"

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_ai_enhanced_contrast_alpha12_aligned"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_scl_2"

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_60_flicker_dev6"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_half_all"

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_test1"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_half_all_merged"
    
    
    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_24cams_0"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_half_24cams"

    exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_test1_60"
    sequence = "2025-05-20_18-24-20_hamit_burak_1_half_all"

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_scl_1_enhanced_test2"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_new1"

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_60_flicker_dev38_w_001_005"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_half_all"

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_60_flicker_dev40"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_half_all"


    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_60_flicker_dev42"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_half_all"
    
    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_60_flicker_dev62_hsv_01_smooth_knn200"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_half_all"

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_60_flicker_dev67"
    # sequence = "2025-05-20_18-24-20_hamit_burak_1_half_all"



    exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_test1"
    sequence = "2025-05-20_18-24-20_hamit_burak_1_half_all_merged"

        
    viewer = Viewer(seq=sequence, exp=exp_name,  port=8088, title="Demo1 Spaceport", w=1920, h=1080)
    time.sleep(0.2)
    viewer.start_viewer()
   

