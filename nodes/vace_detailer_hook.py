import torch
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import comfy.utils
from core.utils import wan_print, wan_print_d
from core.vace_encoding import encode_vace_advanced

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
        self.alignment_config = alignment_config
        self.current_seg_index = 0        
        self.current_crop_region = None
        self.original_dimensions = None
        self.latent_crop = None

        if self.segs and len(self.segs) >= 2:
            wan_print(f"Detailer Hook initialized with {len(self.segs[1])} SEGS")

        self.wva_options = self.alignment_config.get('wva_options', None)
        self.debug = self.wva_options.get_option('enable_debug_prints', False) if self.wva_options else False

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
        seg_total = len(self.segs[1])
        seg_index = self.current_seg_index
        
        x1, y1, x2, y2 = seg.crop_region
        
        expected_w = (x2 - x1) // 8
        expected_h = (y2 - y1) // 8
        
        # Calculate upscale factor if dimensions don't match
        upscale_factor_w = latent_w / expected_w if expected_w > 0 else 1.0
        upscale_factor_h = latent_h / expected_h if expected_h > 0 else 1.0
        upscale_factor = (upscale_factor_w + upscale_factor_h) / 2  # Average for consistency

        wan_print(f"[-- SEGS Index: {seg_index} / {seg_total} --]:")
        if expected_w != latent_w or expected_h != latent_h:
            wan_print(f"Dimensions ({expected_w}x{expected_h}) don't match latent ({latent_w}x{latent_h})")
            wan_print(f"Detected upscale factor: {upscale_factor:.4f} (crop: {seg.crop_region})")

            if self.debug:
                wan_print_d(f"    Original crop region: {seg.crop_region}")
                wan_print_d(f"    Expected latent size: {expected_w}x{expected_h}")
                wan_print_d(f"    Actual latent size:   {latent_w}x{latent_h}")
                wan_print_d(f"    Upscale factor:       {upscale_factor:.4f}")
        else:
            upscale_factor = 1.0
            wan_print(f"Region matches dimensions {latent_w}x{latent_h} (crop: {seg.crop_region})")
            wan_print(f"Not upscaling VACE region.")

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
        
        return seg_index, seg, crop_region_latent, upscale_factor
    
    def reset_hook_state(self):
        self.current_seg_index = 0
        wan_print("SEGS index reset to 0")
    
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
    

    def _encode_vace_with_crop(self, wva_pipe, crop_region_pixels, target_width_pixels, target_height_pixels, 
                                      length, batch_size, vae, positive, negative):
        
        if self.debug:
            wan_print_d(f"Performing VACE encoding for upscaled region")
            wan_print_d(f"  Crop region (pixels): {crop_region_pixels}")
            wan_print_d(f"  Target size (pixels): {target_width_pixels}x{target_height_pixels}")

        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region_pixels
        
        control_video_cropped = None
        if wva_pipe.control_video_1 is not None:
            # Crop control video to region
            control_video_cropped = wva_pipe.control_video_1[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
            # Resize to target dimensions
            control_video_cropped = comfy.utils.common_upscale(
                control_video_cropped.movedim(-1, 1), 
                target_width_pixels, target_height_pixels, 
                "bilinear", "center"
            ).movedim(1, -1)
            
            if self.debug:
                wan_print_d(f"  Cropped control video from {wva_pipe.control_video_1.shape} to {control_video_cropped.shape}")

        control_masks_cropped = None
        if wva_pipe.control_masks_1 is not None:
            # Crop masks to region
            if wva_pipe.control_masks_1.ndim == 3:
                control_masks_cropped = wva_pipe.control_masks_1[:, crop_y1:crop_y2, crop_x1:crop_x2]
            elif wva_pipe.control_masks_1.ndim == 4:
                control_masks_cropped = wva_pipe.control_masks_1[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
            
            # Resize to target dimensions
            if control_masks_cropped.ndim == 3:
                control_masks_cropped = control_masks_cropped.unsqueeze(1)
            control_masks_cropped = comfy.utils.common_upscale(
                control_masks_cropped, 
                target_width_pixels, target_height_pixels, 
                "bilinear", "center"
            )
            if control_masks_cropped.ndim == 4 and control_masks_cropped.shape[1] == 1:
                control_masks_cropped = control_masks_cropped.squeeze(1)
        
            if self.debug:
                wan_print_d(f"  Cropped control masks from {wva_pipe.control_masks_1.shape} to {control_masks_cropped.shape}")

        # Call encode_vace_advanced with the cleaned conditioning and processed inputs
        positive_out, negative_out, neg_phant_img_out, latent_out, trim_latent = encode_vace_advanced(
            positive                = positive,
            negative                = negative,
            vae                     = vae,
            width                   = target_width_pixels,
            height                  = target_height_pixels,
            length                  = length,
            batch_size              = batch_size,
            vace_strength_1         = wva_pipe.vace_strength_1,
            vace_strength_2         = wva_pipe.vace_strength_2,
            vace_ref_strength_1     = wva_pipe.vace_ref_strength_1,
            vace_ref_strength_2     = wva_pipe.vace_ref_strength_2,
            control_video_1         = control_video_cropped,
            control_masks_1         = control_masks_cropped,
            vace_reference_1        = wva_pipe.vace_reference_1,
            phantom_images          = wva_pipe.phantom_images,
            phantom_mask_value      = wva_pipe.phantom_mask_value,
            phantom_control_value   = wva_pipe.phantom_control_value,
            phantom_vace_strength   = wva_pipe.phantom_vace_strength,
            wva_options             = wva_pipe.wva_options
        )

        if self.wva_options and self.wva_options.get_option('phantom_combined_negative', True) and neg_phant_img_out is not None:
            negative_out = negative_out + neg_phant_img_out
            if self.debug:
                wan_print_d("Combined Phantom negative into negative tensor")
        
        return positive_out, negative_out
    
    
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
                    padding = (pad_w, 0, pad_h, 0)
                else:  # bottom_right (default)
                    padding = (0, pad_w, 0, pad_h)
                
                samples = torch.nn.functional.pad(samples, padding, mode='constant', value=0)
                
                if self.debug:
                    wan_print_d(f"Padded latent from {w}x{h} to {samples.shape[-1]}x{samples.shape[-2]} using {pad_direction} strategy")
                
                latent_image = latent_image.copy()
                latent_image['samples'] = samples
        
        return latent_image
    
    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        wva_pipe = self.alignment_config.get('wva_pipe')
        
        if not wva_pipe or not wva_pipe.has_minimum_inputs():
            wan_print("WVAPipe not provided or missing inputs - skipping VACE detailer hook")
            return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
                
        if 'samples' not in latent_image:
            raise ValueError("[WanVaceAdvanced] Error: latent_image does not contain 'samples' key. Cannot process VACE frames.")
        
        latent_samples = latent_image['samples']
        
        current_latent_width = latent_samples.shape[-1]
        current_latent_height = latent_samples.shape[-2]
        latent_width = None
        latent_height = None
        
        if wva_pipe.width is not None:
            latent_width = wva_pipe.width // 8

        if wva_pipe.height is not None:
            latent_height = wva_pipe.height // 8

        if latent_width is None or latent_height is None:
            # Determine original dimensions from WVAPipe source data
            if wva_pipe.control_video_1 is not None:
                control_video_shape = wva_pipe.control_video_1.shape
                latent_height = control_video_shape[1] // 8 if latent_height is None else latent_height
                latent_width = control_video_shape[2] // 8 if latent_width is None else latent_width
            elif wva_pipe.control_masks_1 is not None:
                mask_shape = wva_pipe.control_masks_1.shape
                latent_height = mask_shape[1] // 8 if latent_height is None else latent_height
                latent_width = mask_shape[2] // 8 if latent_width is None else latent_width

            if latent_width is None or latent_height is None:
                raise ValueError("[WanVaceAdvanced] Error: Could not determine original latent dimensions. Please provide width and height in WVAPipe or ensure control video/masks are available.")
        
        crop_valid = False
        if hasattr(self, 'latent_crop') and self.latent_crop is not None:
            x1, y1, x2, y2 = self.latent_crop
            crop_h = y2 - y1
            crop_w = x2 - x1
            if crop_h == current_latent_height and crop_w == current_latent_width:
                crop_valid = True
            else:
                wan_print(f"Warning: Stored crop dimensions ({crop_w}x{crop_h}) don't match current latent ({current_latent_width}x{current_latent_height}). Recalculating...")
        
        if not crop_valid:
            result = self._get_current_seg_sequential(current_latent_width, current_latent_height)
            if len(result) == 4:
                seg_index, matched_seg, crop_region_latent, upscale_factor = result
            
            if matched_seg is not None:
                x1, y1, x2, y2 = crop_region_latent
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(latent_width, x2)
                y2 = min(latent_height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    wan_print(f"Error: Invalid crop region after bounds check: ({x1}, {y1}, {x2}, {y2})")
                    wan_print("Using full frame instead")
                    x1, y1, x2, y2 = 0, 0, latent_width, latent_height
                
                if self.debug:
                    wan_print_d(f"Final crop region: ({x1}, {y1}, {x2}, {y2}) -> size: {x2-x1}x{y2-y1}")
                    if x1 == 0 and y1 == 0 and x2 == latent_width and y2 == latent_height:
                        wan_print_d("Note: Using full frame (segment at image bounds)")
            elif current_latent_height == latent_height and current_latent_width == latent_width:
                x1, y1, x2, y2 = 0, 0, latent_width, latent_height
                wan_print(f"Using full frame for latent dimensions {current_latent_width}x{current_latent_height}")
            else:
                raise ValueError(
                    f"[WanVaceAdvanced] Error: Cannot determine crop region for latent size {current_latent_width}x{current_latent_height}. "
                    f"No matching SEG found and dimensions differ from original VACE size ({latent_width}x{latent_height}). "
                    f"Ensure SEGS data is provided to the hook and SEG crop regions match the processed latent dimensions."
                )
        
        latent_image = self._adjust_latent_for_patch_size(latent_image)
        
        adjusted_latent_samples = latent_image['samples']
        adjusted_h = adjusted_latent_samples.shape[-2]
        adjusted_w = adjusted_latent_samples.shape[-1]
        
        if adjusted_w != current_latent_width or adjusted_h != current_latent_height:
            w_diff = adjusted_w - current_latent_width
            h_diff = adjusted_h - current_latent_height
            
            x2 += w_diff
            y2 += h_diff
            
            wan_print(f"Adjusted latent from {current_latent_width}x{current_latent_height} to {adjusted_w}x{adjusted_h}")
            wan_print(f"Updated crop region to ({x1}, {y1}, {x2}, {y2})")
        
        final_latent_samples = latent_image['samples']
        final_latent_h = final_latent_samples.shape[-2]
        final_latent_w = final_latent_samples.shape[-1]
        
        seg_crop_x1, seg_crop_y1, seg_crop_x2, seg_crop_y2 = matched_seg.crop_region
        crop_region_pixels = (seg_crop_x1, seg_crop_y1, seg_crop_x2, seg_crop_y2)
        if self.debug:
            wan_print_d(f"Using WVAPipe for new VACE encoding")

        latent_length = latent_samples.shape[2] if len(latent_samples.shape) > 4 else 1
        length = (latent_length - 1) * 4 + 1
        batch_size = latent_samples.shape[0]
        
        target_width_pixels = final_latent_w * 8
        target_height_pixels = final_latent_h * 8
        
        positive_new, negative_new = self._encode_vace_with_crop(
            wva_pipe, crop_region_pixels, 
            target_width_pixels, target_height_pixels,
            length, batch_size, wva_pipe.vae,
            positive, negative
        )

        # Extract the new VACE data from the new conditioning for debugging
        new_vace_frames, new_vace_masks, new_vace_strength, new_phantom = self._extract_vace_from_conditioning(positive_new)

        if self.debug:
            wan_print_d("\nPre-KSampler tensor verification:")
            wan_print_d(f"  Crop region (latent): ({x1}, {y1}, {x2}, {y2})")
            wan_print_d(f"  Original VACE dimensions: {latent_width}x{latent_height}")
            wan_print_d(f"  Current latent dimensions: {current_latent_width}x{current_latent_height}")

            latent_samples_final = latent_image['samples']
            wan_print_d(f"  Final latent shape: {latent_samples_final.shape}")
            
            if new_vace_frames and len(new_vace_frames) > 0 and new_vace_frames[0] is not None:
                vace_shape = new_vace_frames[0].shape
                wan_print_d(f"  Fresh VACE frames shape: {vace_shape}")

                vace_h, vace_w = vace_shape[-2], vace_shape[-1]
                latent_h, latent_w = latent_samples_final.shape[-2], latent_samples_final.shape[-1]

                wan_print_d(f"  VACE spatial size: {vace_w}x{vace_h}")
                wan_print_d(f"  Latent spatial size: {latent_w}x{latent_h}")
                
                if vace_h != latent_h or vace_w != latent_w:
                    wan_print_d("  WARNING: VACE and latent dimensions don't match!")
                else:
                    wan_print_d("  VACE and latent dimensions match")

                # VACE requires even dimensions
                if vace_h % 2 != 0 or vace_w % 2 != 0:
                    wan_print_d(f"  WARNING: VACE dimensions are not even! {vace_w}x{vace_h}")
                else:
                    wan_print_d("  VACE dimensions are even")
            else:
                wan_print_d("  ERROR: No new VACE frames available!")

            if new_vace_masks and len(new_vace_masks) > 0 and new_vace_masks[0] is not None:
                mask_shape = new_vace_masks[0].shape
                wan_print_d(f"  New VACE masks shape: {mask_shape}")
            else:
                wan_print_d("  No VACE masks present")
            
            if new_phantom is not None:
                wan_print_d("  Phantom frames processed via new encoding")
            else:
                wan_print_d("  No Phantom frames present")

            wan_print_d("End tensor verification\n")

        return model, seed, steps, cfg, sampler_name, scheduler, positive_new, negative_new, latent_image, denoise


class VACEAdvDetailerHookProvider:   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
            },
            "optional": {
                "wva_pipe": ("WVA_PIPE", {}),
                "wva_options": ("WVA_OPTIONS", {}),
            }
        }
    
    RETURN_TYPES = ("DETAILER_HOOK",)
    FUNCTION = "create_hook"
    CATEGORY = "WanVaceAdvanced/Hooks"

    def create_hook(self, segs, wva_pipe=None, wva_options=None, latent_pad_direction="bottom_right", crop_calculation_from="bottom_right"):        
        if not IMPACT_PACK_AVAILABLE:
            raise RuntimeError("Impact Pack is required for VACE DetailerHook functionality. Please install ComfyUI-Impact-Pack.")
        
        if wva_options is not None:
            if wva_pipe.wva_options is not None:
                wan_print("Warning: Overriding WVAPipe WVAOptions with DetailerHook WVAOptions")
            wva_pipe.wva_options = wva_options
        
        alignment_config = {
            'latent_pad_direction': latent_pad_direction,
            'crop_calculation_from': crop_calculation_from,
            'wva_pipe': wva_pipe,
            'wva_options': wva_options,
        }

        hook = VACEAdvDetailerHook(segs=segs, alignment_config=alignment_config)
        return (hook,)


if IMPACT_PACK_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "VACEAdvDetailerHookProvider": VACEAdvDetailerHookProvider,
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "VACEAdvDetailerHookProvider": "VACEAdvancedDetailerHook",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}