DEBUG_ENABLED = True

def wan_print(*args, debug=False, **kwargs):
    if 'wva_options' in kwargs:
        wva_options = kwargs['wva_options']
        if debug:
            if not getattr(wva_options, 'enable_debug_prints', True):
                return
    elif debug and not DEBUG_ENABLED:
        return
    
    if args:
        if not debug:
            first_arg = str(args[0])
            if not first_arg.startswith('[WanVaceAdvanced]'):
                # Don't prepend tag if another tag is present or if it starts with a new line
                if first_arg.startswith('[') or first_arg.startswith('\n'):
                    pass
                args = (f'[WanVaceAdvanced] {first_arg}',) + args[1:]
        else:
            first_arg = str(args[0])
            if not first_arg.startswith('[WanVaceAdvDEBUG]'):
                args = (f'[WanVaceAdvDEBUG] {first_arg}',) + args[1:]

    print(*args, **kwargs)

def wan_print_d(*args, **kwargs):
    wan_print(*args, debug=True, **kwargs)

class WVAOptions:
    def __init__(self,
                 use_tiled_vae=False,
                 enable_debug_prints=True,
                 debug_save_images=False,
                 phantom_resize_mode="center",
                 phantom_combined_negative=False
                ):
        
        self.use_tiled_vae = use_tiled_vae
        self.enable_debug_prints = enable_debug_prints
        self.debug_save_images = debug_save_images
        self.phantom_resize_mode = phantom_resize_mode
        self.phantom_combined_negative = phantom_combined_negative

    def get_option(self, key, default=None):
        return getattr(self, key, default)
    
    def set_option(self, key, value):
        setattr(self, key, value)
        return self
    
    def __str__(self):
        output = "WVAOptions:("
        for key, value in self.__dict__.items():
            output += f"{key}:{value}, "
        output += ")"
        return output

class WVAPipe:
    def __init__(self, width=None, height=None, length=None, batch_size=1,
                 control_video_1=None, control_masks_1=None, vace_reference_1=None,
                 control_video_2=None, control_masks_2=None, vace_reference_2=None,
                 vace_strength_1=1.0, vace_strength_2=1.0, 
                 vace_ref_strength_1=None, vace_ref_strength_2=None,
                 phantom_images=None, phantom_mask_value=1.0, 
                 phantom_control_value=0.0, phantom_vace_strength=0.0,
                 wva_options=None, vae=None):
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.control_video_1 = control_video_1
        self.control_masks_1 = control_masks_1
        self.vace_reference_1 = vace_reference_1
        self.control_video_2 = control_video_2
        self.control_masks_2 = control_masks_2
        self.vace_reference_2 = vace_reference_2
        self.vace_strength_1 = vace_strength_1
        self.vace_strength_2 = vace_strength_2
        self.vace_ref_strength_1 = vace_ref_strength_1
        self.vace_ref_strength_2 = vace_ref_strength_2
        self.phantom_images = phantom_images
        self.phantom_mask_value = phantom_mask_value
        self.phantom_control_value = phantom_control_value
        self.phantom_vace_strength = phantom_vace_strength
        self.wva_options = wva_options
        self.vae = vae
    
    def has_minimum_inputs(self):
        # Check if pipe has at least the minimum inputs for encode_vace_advanced.
        will_work = self.control_video_1 is not None or \
                    self.control_masks_1 is not None or \
                    self.vace_reference_1 is not None or \
                    self.phantom_images is not None
        
        will_work &= self.vae is not None

        return will_work
    
import torch
import comfy.utils

def resize_with_edge_padding(image, target_width, target_height):
    """
    Resize image with edge padding to preserve aspect ratio and avoid cropping.
    
    Based on KJNodes ImageResizeKJv2 and ImagePadKJ implementation:
    - https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/image_nodes.py
    """
    if image is None:
        return None
        
    B, H, W, C = image.shape
    
    if target_width == W and target_height == H:
        return image
    
    # Calculate aspect ratios
    original_aspect = W / H
    target_aspect = target_width / target_height
    
    if original_aspect > target_aspect:
        new_width = target_width
        new_height = int(target_width / original_aspect)
    else:
        new_width = int(target_height * original_aspect)
        new_height = target_height
    
    # First resize to preserve aspect ratio
    # Convert to [frames, channels, height, width] for common_upscale
    image_chw = image.movedim(-1, 1)
    resized_image = comfy.utils.common_upscale(
        image_chw, new_width, new_height, "lanczos", "disabled"
    ).movedim(1, -1)
    
    # Calculate padding needed
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        return resized_image
    
    # Apply edge padding (based on KJNodes ImagePadKJ edge mode)
    padded_image = torch.zeros((B, target_height, target_width, C), 
                              dtype=image.dtype, device=image.device)
    
    for b in range(B):
        resized_b = resized_image[b]  # [height, width, channels]
        
        # Get edge colors for padding
        if new_height > 0 and new_width > 0:
            top_color = resized_b[0, :, :].mean(dim=0)
            bottom_color = resized_b[new_height - 1, :, :].mean(dim=0)
            left_color = resized_b[:, 0, :].mean(dim=0)
            right_color = resized_b[:, new_width - 1, :].mean(dim=0)

            if pad_top > 0:
                padded_image[b, :pad_top, :, :] = top_color
            if pad_bottom > 0:
                padded_image[b, pad_top + new_height:, :, :] = bottom_color
            if pad_left > 0:
                padded_image[b, :, :pad_left, :] = left_color
            if pad_right > 0:
                padded_image[b, :, pad_left + new_width:, :] = right_color
        
        # Place the resized image in the center
        padded_image[b, pad_top:pad_top+new_height, pad_left:pad_left+new_width, :] = resized_b
    
    return padded_image

def debug_save_images(image_frames, prefix="phantom", seg_index=None):
    if image_frames is None:
        return
        
    try:
        import os
        from PIL import Image
        import numpy as np
        import folder_paths
        
        temp_dir = os.path.join(folder_paths.get_temp_directory(), "vace_advanced_debug")
        os.makedirs(temp_dir, exist_ok=True)
        
        num_frames = image_frames.shape[0]
        
        for frame_idx in range(num_frames):
            frame_tensor = image_frames[frame_idx]
            img_np = frame_tensor.detach().cpu().numpy()
            img_np = np.clip(img_np, 0.0, 1.0)
            img_np = (img_np * 255).astype(np.uint8)
            seg_str = f"_seg{seg_index}" if seg_index is not None else ""
            base_filename = f"{prefix}_frame{frame_idx:03d}{seg_str}"
            
            counter = 0
            filename = f"{base_filename}.png"
            filepath = os.path.join(temp_dir, filename)
            
            while os.path.exists(filepath):
                counter += 1
                filename = f"{base_filename}_{counter:03d}.png"
                filepath = os.path.join(temp_dir, filename)
            
            img = Image.fromarray(img_np)
            img.save(filepath)
        
        wan_print(f"Saved {num_frames} {prefix} debug images to {temp_dir}")
        
    except Exception as e:
        wan_print(f"Error saving {prefix} debug images: {e}")

