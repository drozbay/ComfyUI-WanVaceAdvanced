import torch
import sys
from pathlib import Path
import numpy as np
import os
import tempfile
from PIL import Image

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import comfy.model_management
import node_helpers
from core.utils import wan_print

IMPACT_PACK_AVAILABLE = False
DetailerHook = None

try:
    custom_nodes_dir = Path(__file__).parent.parent.parent
    impact_pack_path = None
    
    for folder_name in ['ComfyUI-Impact-Pack', 'comfyui-impact-pack']:
        potential_path = custom_nodes_dir / folder_name
        if potential_path.exists():
            impact_pack_path = potential_path
            break
    
    if impact_pack_path:
        impact_module_path = impact_pack_path / 'modules'
        if str(impact_module_path) not in sys.path:
            sys.path.insert(0, str(impact_module_path))
        
        from impact.hooks import DetailerHook
        IMPACT_PACK_AVAILABLE = True
        wan_print("Impact Pack integration enabled for VACE DetailerHook")
    else:
        wan_print("Impact Pack not found - VACE DetailerHook will not be available")
        
except Exception as e:
    wan_print(f"Could not import Impact Pack hooks: {e}")
    wan_print("VACE DetailerHook will not be available")


class VACEAdvDetailerHook(DetailerHook if DetailerHook else object):    
    def __init__(self, segs=None, alignment_config=None):
        if DetailerHook:
            super().__init__()
        
        self.segs = segs
        self.alignment_config = alignment_config or {
            'latent_pad_direction': 'bottom_right',
            'enable_debug_prints': True,
        }
        
        
        self.current_seg_index = 0
        
        if self.alignment_config.get('enable_debug_prints', True) and self.segs and len(self.segs) >= 2:
            wan_print(f"Hook initialized with {len(self.segs[1])} SEGS, starting at index 0", debug=True)
        
        self.current_crop_region = None
        self.original_dimensions = None
        self.latent_crop = None
    
    def _ensure_vace_compatible_dimensions(self, x1, y1, x2, y2, max_w, max_h):
        width = x2 - x1
        height = y2 - y1
        
        if width % 2 != 0 or height % 2 != 0:
            new_width = width + (width % 2)
            new_height = height + (height % 2)
            
            if width % 2 != 0:
                if x2 < max_w:
                    x2 += 1
                elif x1 > 0:
                    x1 -= 1
                else:
                    wan_print("Warning: Cannot adjust width to even - at boundaries")
            
            if height % 2 != 0:
                if y2 < max_h:
                    y2 += 1
                elif y1 > 0:
                    y1 -= 1
                else:
                    wan_print("Warning: Cannot adjust height to even - at boundaries")
            
            final_width = x2 - x1
            final_height = y2 - y1
            
            if final_width % 2 != 0 or final_height % 2 != 0:
                wan_print(f"Warning: Could not achieve even dimensions ({final_width}x{final_height})")
        
        return x1, y1, x2, y2
    
    def _get_current_seg_sequential(self, latent_w, latent_h):
        if not self.segs or len(self.segs) < 2 or len(self.segs[1]) == 0:
            return None, None, None
        
        if self.current_seg_index >= len(self.segs[1]):
            self.current_seg_index = 0
            if self.alignment_config.get('enable_debug_prints', True):
                wan_print("Wrapped SEG index back to 0", debug=True)
        
        seg = self.segs[1][self.current_seg_index]
        seg_index = self.current_seg_index
        
        x1, y1, x2, y2 = seg.crop_region
        
        expected_w = (x2 - x1) // 8
        expected_h = (y2 - y1) // 8
        
        if expected_w != latent_w or expected_h != latent_h:
            wan_print(f"WARNING: SEG {seg_index} dimensions ({expected_w}x{expected_h}) don't match latent ({latent_w}x{latent_h})")
            wan_print(f"Using SEG {seg_index} anyway (crop: {seg.crop_region})")
        else:
            if self.alignment_config.get('enable_debug_prints', True):
                wan_print(f"Using SEG {seg_index} with matching dimensions {latent_w}x{latent_h} (crop: {seg.crop_region})", debug=True)
        
        calc_from = self.alignment_config.get('crop_calculation_from', 'top_left')
        
        if calc_from == 'bottom_right':
            x2_latent = x2 // 8
            y2_latent = y2 // 8
            x1_latent = x2_latent - latent_w
            y1_latent = y2_latent - latent_h
        else:
            x1_latent = x1 // 8
            y1_latent = y1 // 8
            x2_latent = x1_latent + latent_w
            y2_latent = y1_latent + latent_h
        
        crop_region_latent = (x1_latent, y1_latent, x2_latent, y2_latent)
        
        self.current_seg_index += 1
        
        return seg_index, seg, crop_region_latent
    
    def reset_hook_state(self):
        self.current_seg_index = 0
        
        if self.alignment_config.get('enable_debug_prints', True):
            wan_print("Hook reset for new SEGS batch - starting at index 0", debug=True)
    
    def post_crop_region(self, w, h, item_bbox, crop_region):
        x1, y1, x2, y2 = crop_region
        
        self.original_dimensions = (w, h)
        
        width = x2 - x1
        height = y2 - y1
        
        if width % 16 != 0 or height % 16 != 0:
            new_width = ((width + 15) // 16) * 16
            new_height = ((height + 15) // 16) * 16
            
            width_diff = new_width - width
            height_diff = new_height - height
            
            x1 = max(0, x1 - width_diff // 2)
            x2 = min(w, x1 + new_width)
            if x2 - x1 < new_width:
                x1 = max(0, x2 - new_width)
            
            y1 = max(0, y1 - height_diff // 2)
            y2 = min(h, y1 + new_height)
            if y2 - y1 < new_height:
                y1 = max(0, y2 - new_height)
            
            crop_region = (x1, y1, x2, y2)
        
        self.current_crop_region = crop_region
        x1_latent = crop_region[0] // 8
        y1_latent = crop_region[1] // 8
        x2_latent = crop_region[2] // 8
        y2_latent = crop_region[3] // 8
        
        x1_latent, y1_latent, x2_latent, y2_latent = self._ensure_vace_compatible_dimensions(
            x1_latent, y1_latent, x2_latent, y2_latent, w // 8, h // 8
        )
        
        self.latent_crop = [x1_latent, y1_latent, x2_latent, y2_latent]
        
        return crop_region
    
    def _extract_vace_from_conditioning(self, conditioning):
        if not conditioning or len(conditioning) == 0:
            return None, None, None, None
        
        cond_dict = conditioning[0][1] if len(conditioning[0]) > 1 else {}
        
        vace_frames = cond_dict.get('vace_frames', None)
        vace_mask = cond_dict.get('vace_mask', None)
        vace_strength = cond_dict.get('vace_strength', None)
        phantom_frames = cond_dict.get('time_dim_concat', None)
        
        return vace_frames, vace_mask, vace_strength, phantom_frames
    
    def _crop_vace_frames_spatial(self, vace_frames, y1, y2, x1, x2):
        """Crop VACE frames spatially to SEG region."""
        if vace_frames is None:
            return None
        
        cropped_frames = []
        for frames in vace_frames:
            if frames is None:
                cropped_frames.append(None)
                continue
            
            cropped = frames[:, :, :, y1:y2, x1:x2]
            cropped_frames.append(cropped)
        
        return cropped_frames
    
    def _crop_vace_masks_spatial(self, vace_masks, y1, y2, x1, x2):
        """Crop VACE masks spatially to SEG region."""
        if vace_masks is None:
            return None
        
        cropped_masks = []
        for mask in vace_masks:
            if mask is None:
                cropped_masks.append(None)
                continue
            
            cropped = mask[:, :, :, y1:y2, x1:x2]
            cropped_masks.append(cropped)
        
        return cropped_masks
    
    
    def _decode_phantom_frames(self, phantom_frames):
        if phantom_frames is None:
            return None
            
        vae = self.alignment_config.get('debug_vae')
        if vae is None:
            if self.alignment_config.get('enable_debug_prints', True):
                wan_print("Warning: Cannot decode Phantom frames - no VAE provided")
            return None
            
        try:
            decoded_images = []
            
            for frame_idx in range(phantom_frames.shape[2]):
                frame_latent = phantom_frames[:, :, frame_idx:frame_idx+1, :, :]
                
                if self.alignment_config.get('enable_debug_prints', True) and frame_idx == 0:
                    wan_print(f"Decoding latent shape: {frame_latent.shape}", debug=True)
                
                decoded = vae.decode(frame_latent)
                if self.alignment_config.get('enable_debug_prints', True) and frame_idx == 0:
                    wan_print(f"Decoded shape: {decoded.shape}", debug=True)
                
                img_tensor = decoded[0]
                img_np = img_tensor.detach().cpu().numpy()
                
                if self.alignment_config.get('enable_debug_prints', True) and frame_idx == 0:
                    wan_print(f"Numpy shape before conversion: {img_np.shape}", debug=True)
                
                if len(img_np.shape) == 3:
                    if img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))
                elif len(img_np.shape) == 4:
                    img_np = img_np[0] if img_np.shape[0] == 1 else img_np
                    if img_np.shape[-1] != 3 and img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))
                
                if len(img_np.shape) != 3 or img_np.shape[2] != 3:
                    if self.alignment_config.get('enable_debug_prints', True):
                        wan_print(f"Warning: Skipping frame {frame_idx} with unexpected shape: {img_np.shape}", debug=True)
                    continue
                
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(img_np)
                decoded_images.append(pil_image)
            
            if self.alignment_config.get('enable_debug_prints', True):
                wan_print(f"Decoded {len(decoded_images)} Phantom frames", debug=True)
            
            return decoded_images
            
        except Exception as e:
            if self.alignment_config.get('enable_debug_prints', True):
                wan_print(f"Error decoding Phantom frames: {e}", debug=True)
            return None
    
    def _resize_and_encode_phantom_images(self, decoded_images, target_h_pixels, target_w_pixels):
        if decoded_images is None:
            return None
            
        vae = self.alignment_config.get('debug_vae')
        if vae is None:
            return None
            
        try:
            resized_latents = []
            
            for frame_idx, pil_image in enumerate(decoded_images):
                resized_image = pil_image.resize((target_w_pixels, target_h_pixels), Image.LANCZOS)
                
                img_np = np.array(resized_image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                
                # Debug: Check tensor shape before encoding
                if self.alignment_config.get('enable_debug_prints', True):
                    wan_print(f"Encoding tensor shape: {img_tensor.shape}", debug=True)
                
                # Ensure tensor is in the right format and device
                if len(img_tensor.shape) != 4 or img_tensor.shape[3] != 3:
                    raise ValueError(f"Expected tensor shape [1, height, width, 3], got {img_tensor.shape}")
                
                if hasattr(vae, 'device'):
                    img_tensor = img_tensor.to(vae.device)
                elif hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'device'):
                    img_tensor = img_tensor.to(vae.first_stage_model.device)
                else:
                    for param in vae.parameters():
                        img_tensor = img_tensor.to(param.device)
                        break
                
                try:
                    encoded_latent = vae.encode(img_tensor)
                    if isinstance(encoded_latent, dict):
                        encoded_latent = encoded_latent['samples']
                    elif hasattr(encoded_latent, 'sample'):
                        encoded_latent = encoded_latent.sample()
                except Exception as encode_error:
                    if self.alignment_config.get('enable_debug_prints', True):
                        wan_print(f"VAE encode failed: {encode_error}", debug=True)
                        wan_print(f"Input tensor shape: {img_tensor.shape}", debug=True)
                        wan_print(f"Input tensor device: {img_tensor.device}", debug=True)
                        wan_print(f"Input tensor dtype: {img_tensor.dtype}", debug=True)
                    raise
                resized_latents.append(encoded_latent)
            
            if resized_latents:
                # Concatenate along frame dimension (dim=2)
                phantom_latent = torch.cat(resized_latents, dim=2)
                
                if self.alignment_config.get('enable_debug_prints', True):
                    wan_print(f"Resized and re-encoded Phantom to {target_w_pixels}x{target_h_pixels} pixels", debug=True)
                
                return phantom_latent
            
        except Exception as e:
            if self.alignment_config.get('enable_debug_prints', True):
                wan_print(f"Error resizing and encoding Phantom frames: {e}", debug=True)
        
        return None
    
    def _debug_save_phantom_frames(self, phantom_frames, prefix="phantom", seg_index=None):
        """
        Debug helper: decode and save Phantom frames to temporary folder for visual inspection.
        """
        if not self.alignment_config.get('debug_save_phantom', False):
            return
            
        vae = self.alignment_config.get('debug_vae')
        if vae is None:
            wan_print("Warning: debug_save_phantom enabled but no VAE provided")
            return
            
        if phantom_frames is None:
            return
            
        try:
            temp_dir = os.path.join(tempfile.gettempdir(), "vace_phantom_debug")
            os.makedirs(temp_dir, exist_ok=True)
            
            for frame_idx in range(phantom_frames.shape[2]): 
                frame_latent = phantom_frames[:, :, frame_idx:frame_idx+1, :, :]  # Keep frame dim

                decoded = vae.decode(frame_latent)
                
                img_tensor = decoded[0] 
                img_np = img_tensor.detach().cpu().numpy()
                
                if len(img_np.shape) == 3:
                    if img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))
                elif len(img_np.shape) == 4:
                    img_np = img_np[0] if img_np.shape[0] == 1 else img_np
                    if img_np.shape[-1] != 3 and img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))
                
                if len(img_np.shape) != 3 or img_np.shape[2] != 3:
                    wan_print(f"Warning: Unexpected decoded Phantom image shape: {img_np.shape}")
                    continue
                
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                
                seg_str = f"_seg{seg_index}" if seg_index is not None else ""
                filename = f"{prefix}_frame{frame_idx:03d}{seg_str}.png"
                filepath = os.path.join(temp_dir, filename)
                
                img = Image.fromarray(img_np)
                img.save(filepath)
            
            wan_print(f"Saved {phantom_frames.shape[2]} Phantom debug frames to {temp_dir}")
            
        except Exception as e:
            wan_print(f"Error saving Phantom debug frames: {e}")
    
    def _adjust_latent_for_patch_size(self, latent_image):
        if 'samples' not in latent_image:
            return latent_image
        
        samples = latent_image['samples']
        
        h, w = samples.shape[-2], samples.shape[-1]
        
        if h % 2 != 0 or w % 2 != 0:
            # Pad to even dimensions
            pad_h = (2 - h % 2) % 2
            pad_w = (2 - w % 2) % 2
            
            if pad_h > 0 or pad_w > 0:
                pad_direction = self.alignment_config.get('latent_pad_direction', 'bottom_right')
                
                if pad_direction == 'top_left':
                    # Pad on the left and top
                    padding = (pad_w, 0, pad_h, 0)
                else:  # bottom_right (default)
                    # Pad on the right and bottom
                    padding = (0, pad_w, 0, pad_h)
                
                samples = torch.nn.functional.pad(samples, padding, mode='constant', value=0)
                
                if self.alignment_config.get('enable_debug_prints', True):
                    wan_print(f"Padded latent from {w}x{h} to {samples.shape[-1]}x{samples.shape[-2]} using {pad_direction} strategy", debug=True)
                
                latent_image = latent_image.copy()
                latent_image['samples'] = samples
        
        return latent_image
    
    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):       
        vace_frames, vace_masks, vace_strength, phantom_frames = self._extract_vace_from_conditioning(positive)
        
        if vace_frames is None:
            return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
        
        if 'samples' not in latent_image:
            raise ValueError("[WanVaceAdvanced] Error: latent_image does not contain 'samples' key. Cannot process VACE frames.")
        
        latent_samples = latent_image['samples']
        
        current_h = latent_samples.shape[-2]
        current_w = latent_samples.shape[-1]
        
        if not vace_frames or len(vace_frames) == 0:
            raise ValueError("[WanVaceAdvanced] Error: VACE frames list is empty. Cannot determine original dimensions.")
        
        original_vace = vace_frames[0]
        if original_vace is None:
            raise ValueError("[WanVaceAdvanced] Error: First VACE frame tensor is None. Cannot process.")
        
        original_h = original_vace.shape[-2]
        original_w = original_vace.shape[-1]
        
        crop_valid = False
        if hasattr(self, 'latent_crop') and self.latent_crop is not None:
            x1, y1, x2, y2 = self.latent_crop
            crop_h = y2 - y1
            crop_w = x2 - x1
            if crop_h == current_h and crop_w == current_w:
                crop_valid = True
            else:
                wan_print(f"Warning: Stored crop dimensions ({crop_w}x{crop_h}) don't match current latent ({current_w}x{current_h}). Recalculating...")
        
        if not crop_valid:
            seg_index, matched_seg, crop_region_latent = self._get_current_seg_sequential(current_w, current_h)
            
            if matched_seg is not None:
                x1, y1, x2, y2 = crop_region_latent
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_w, x2)
                y2 = min(original_h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    wan_print(f"Error: Invalid crop region after bounds check: ({x1}, {y1}, {x2}, {y2})")
                    wan_print("Using full frame instead")
                    x1, y1, x2, y2 = 0, 0, original_w, original_h
                
                if self.alignment_config.get('enable_debug_prints', True):
                    wan_print(f"Final crop region: ({x1}, {y1}, {x2}, {y2}) -> size: {x2-x1}x{y2-y1}", debug=True)
                    if x1 == 0 and y1 == 0 and x2 == original_w and y2 == original_h:
                        wan_print("Note: Using full frame (segment at image bounds)", debug=True)
            elif current_h == original_h and current_w == original_w:
                x1, y1, x2, y2 = 0, 0, original_w, original_h
                wan_print(f"Using full frame for latent dimensions {current_w}x{current_h}")
            else:
                raise ValueError(
                    f"[WanVaceAdvanced] Error: Cannot determine crop region for latent size {current_w}x{current_h}. "
                    f"No matching SEG found and dimensions differ from original VACE size ({original_w}x{original_h}). "
                    f"Ensure SEGS data is provided to the hook and SEG crop regions match the processed latent dimensions."
                )
        
        latent_image = self._adjust_latent_for_patch_size(latent_image)
        
        adjusted_latent_samples = latent_image['samples']
        adjusted_h = adjusted_latent_samples.shape[-2]
        adjusted_w = adjusted_latent_samples.shape[-1]
        
        if adjusted_w != current_w or adjusted_h != current_h:
            w_diff = adjusted_w - current_w
            h_diff = adjusted_h - current_h
            
            x2 += w_diff
            y2 += h_diff
            
            wan_print(f"Adjusted latent from {current_w}x{current_h} to {adjusted_w}x{adjusted_h}")
            wan_print(f"Updated crop region to ({x1}, {y1}, {x2}, {y2})")
        
        cropped_frames = self._crop_vace_frames_spatial(vace_frames, y1, y2, x1, x2)
        cropped_masks = self._crop_vace_masks_spatial(vace_masks, y1, y2, x1, x2)
        
        resized_phantom = None
        if phantom_frames is not None:
            vae = self.alignment_config.get('debug_vae')
            if vae is None:
                raise ValueError(
                    "[WanVaceAdvanced] Error: Phantom frames detected but no VAE provided. "
                    "A VAE is required to process Phantom frames. Please connect a VAE to the 'vae' input."
                )
            
            decoded_images = self._decode_phantom_frames(phantom_frames)
            
            if decoded_images is not None:
                final_latent_samples = latent_image['samples']
                final_latent_h = final_latent_samples.shape[-2]
                final_latent_w = final_latent_samples.shape[-1]
                
                # Convert to pixel dimensions (multiply by 8 for VAE)
                target_h_pixels = final_latent_h * 8
                target_w_pixels = final_latent_w * 8
                
                resized_phantom = self._resize_and_encode_phantom_images(decoded_images, target_h_pixels, target_w_pixels)
                
                if hasattr(self, 'current_seg_index'):
                    seg_idx = self.current_seg_index
                else:
                    seg_idx = None
                self._debug_save_phantom_frames(resized_phantom, "resized", seg_idx)
        
        vace_dict = {
            'vace_frames': cropped_frames,
            'vace_mask': cropped_masks,
            'vace_strength': vace_strength
        }
        
        if resized_phantom is not None:
            vace_dict['time_dim_concat'] = resized_phantom
        
        import copy
        positive_copy = copy.deepcopy(positive)
        negative_copy = copy.deepcopy(negative)
        
        positive_copy = node_helpers.conditioning_set_values(positive_copy, vace_dict, append=False)
        negative_copy = node_helpers.conditioning_set_values(negative_copy, vace_dict, append=False)
        
        if self.alignment_config.get('enable_debug_prints', True):
            wan_print("\nPre-KSampler tensor verification:", debug=True)
            wan_print(f"  Crop region (latent): ({x1}, {y1}, {x2}, {y2})", debug=True)
            wan_print(f"  Original VACE dimensions: {original_w}x{original_h}", debug=True)
            wan_print(f"  Current latent dimensions: {current_w}x{current_h}", debug=True)
            
            latent_samples_final = latent_image['samples']
            wan_print(f"  Final latent shape: {latent_samples_final.shape}", debug=True)
            
            if cropped_frames and len(cropped_frames) > 0 and cropped_frames[0] is not None:
                vace_shape = cropped_frames[0].shape
                wan_print(f"  Cropped VACE frames shape: {vace_shape}", debug=True)
                
                vace_h, vace_w = vace_shape[-2], vace_shape[-1]
                latent_h, latent_w = latent_samples_final.shape[-2], latent_samples_final.shape[-1]
                
                wan_print(f"  VACE spatial size: {vace_w}x{vace_h}", debug=True)
                wan_print(f"  Latent spatial size: {latent_w}x{latent_h}", debug=True)
                
                if vace_h != latent_h or vace_w != latent_w:
                    wan_print("  WARNING: VACE and latent dimensions don't match!", debug=True)
                else:
                    wan_print("  ✓ VACE and latent dimensions match", debug=True)
                
                # VACE requires even dimensions
                if vace_h % 2 != 0 or vace_w % 2 != 0:
                    wan_print(f"  WARNING: VACE dimensions are not even! {vace_w}x{vace_h}", debug=True)
                else:
                    wan_print("  ✓ VACE dimensions are even", debug=True)
            else:
                wan_print("  ERROR: No cropped VACE frames available!", debug=True)
            
            if cropped_masks and len(cropped_masks) > 0 and cropped_masks[0] is not None:
                mask_shape = cropped_masks[0].shape
                wan_print(f"  Cropped VACE masks shape: {mask_shape}", debug=True)
            else:
                wan_print("  No VACE masks present", debug=True)
            
            if resized_phantom is not None:
                phantom_shape = resized_phantom.shape
                wan_print(f"  Resized Phantom frames shape: {phantom_shape}", debug=True)
                
                phantom_h, phantom_w = phantom_shape[-2], phantom_shape[-1]
                if phantom_h != latent_h or phantom_w != latent_w:
                    wan_print("  WARNING: Phantom and latent dimensions don't match!", debug=True)
                else:
                    wan_print("  ✓ Phantom and latent dimensions match", debug=True)
            else:
                wan_print("  No Phantom frames present", debug=True)
            
            wan_print("End tensor verification\n", debug=True)
        
        return model, seed, steps, cfg, sampler_name, scheduler, positive_copy, negative_copy, latent_image, denoise


class VACEAdvDetailerHookProvider:   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
            },
            "optional": {
                "latent_pad_direction": (["bottom_right", "top_left"], {"default": "bottom_right", "tooltip": "Direction to pad latent when adjusting for even dimensions"}),
                "crop_calculation_from": (["top_left", "bottom_right"], {"default": "bottom_right", "tooltip": "Calculate crop region from top-left or bottom-right corner"}),
                "enable_debug_prints": ("BOOLEAN", {"default": True, "tooltip": "Enable detailed debugging output"}),
                "debug_save_phantom": ("BOOLEAN", {"default": False, "tooltip": "Save decoded Phantom frames to temp folder for debugging"}),
                "vae": ("VAE", {"tooltip": "VAE for decoding Phantom frames (required if debug_save_phantom is True)"}),
            }
        }
    
    RETURN_TYPES = ("DETAILER_HOOK",)
    FUNCTION = "create_hook"
    CATEGORY = "WanVaceAdvanced/Hooks"

    def create_hook(self, segs, latent_pad_direction="bottom_right", crop_calculation_from="bottom_right", enable_debug_prints=True, debug_save_phantom=False, vae=None):
        """Create a VACE DetailerHook that works with existing conditioning."""
        
        if not IMPACT_PACK_AVAILABLE:
            raise RuntimeError("Impact Pack is required for VACE DetailerHook functionality. Please install ComfyUI-Impact-Pack.")
        
        alignment_config = {
            'latent_pad_direction': latent_pad_direction,
            'crop_calculation_from': crop_calculation_from,
            'enable_debug_prints': enable_debug_prints,
            'debug_save_phantom': debug_save_phantom,
            'debug_vae': vae,
        }
        
        hook = VACEAdvDetailerHook(segs=segs, alignment_config=alignment_config)
        return (hook,)


if IMPACT_PACK_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "VACEAdvDetailerHookProvider": VACEAdvDetailerHookProvider,
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "VACEAdvDetailerHookProvider": "VACE Advanced Detailer Hook",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}