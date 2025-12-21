import comfy.model_management
import comfy.model_patcher
import comfy.conds
import logging
import types

import torch
import comfy.ldm.wan.model
from comfy.ldm.wan.model import sinusoidal_embedding_1d
from .utils import wan_print

# List of custom conditioning keys that should be passed through to model_conds
CUSTOM_PASSTHROUGH_KEYS = [
    "window_reference_batch",   # Per-window reference images for context windows
    "window_reference_mapping", # Explicit mapping string for window-to-reference assignment
]

def _patch_extra_conds_passthrough(model_obj):
    """
    Patch extra_conds on a model to pass through custom conditioning keys.
    """
    if hasattr(model_obj, '_wva_extra_conds_patched'):
        return

    original_extra_conds = model_obj.extra_conds

    def patched_extra_conds(self, **kwargs):
        out = original_extra_conds(**kwargs)

        for key in CUSTOM_PASSTHROUGH_KEYS:
            value = kwargs.get(key, None)
            if value is not None:
                # Wrap in CONDConstant so it survives the conditioning processing
                out[key] = comfy.conds.CONDConstant(value)

        return out

    model_obj.extra_conds = types.MethodType(patched_extra_conds, model_obj)
    model_obj._wva_extra_conds_patched = True
    logging.debug("WanVaceAdvanced: extra_conds passthrough patch applied")


def wrap_vace_phantom_wan_model(model):
    """
    Patch a VaceWanModel to support per-frame strength and Phantom embeds.
    Based on the WanVideoTeaCacheKJ implementation in comfyui-kjnodes.
    Note: The model should already be cloned before passing to this function.
    """
    
    # Check if this is a VaceWanModel
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, comfy.ldm.wan.model.VaceWanModel):
        logging.info("Not a VaceWanModel, skipping per-frame strength patch")
        return model

    # Patch extra_conds to pass through custom conditioning keys
    _patch_extra_conds_passthrough(model.model)

    # Register context windows callbacks if context windows are enabled
    context_handler = model.model_options.get("context_handler")
    if context_handler is not None:
        from .context_windows_hook import register_vace_context_windows_callbacks
        register_vace_context_windows_callbacks(model, context_handler)

    def unet_wrapper_function(model_function, kwargs):
        # Extract parameters from kwargs
        input_data = kwargs["input"]
        timestep = kwargs["timestep"] 
        c = kwargs["c"]
        
        vace_strength = c.get("vace_strength", [1.0])

        # Check if we have nested lists (our separate reference strength format)
        use_patched_forward_orig = (
            isinstance(vace_strength, list) and 
            len(vace_strength) > 0 and
            isinstance(vace_strength[0], list)
        )

        # Also check if we have Phantom concats (c.get("time_dim_concat") exists and is a tensor)
        use_patched_forward_orig = use_patched_forward_orig or (
            c.get("time_dim_concat") is not None and
            isinstance(c.get("time_dim_concat"), torch.Tensor)
        )

        if use_patched_forward_orig:
            # Extract frame count information from conditioning
            phantom_frames = 0
            reference_frames = 0
            
            # Check for phantom frames from time_dim_concat
            time_dim_concat = c.get("time_dim_concat")
            if time_dim_concat is not None and isinstance(time_dim_concat, torch.Tensor):
                phantom_frames = time_dim_concat.shape[2]  # temporal dimension
            
            # Check for reference frames from c['vace_context']
            vace_context = c.get("vace_context")
            if vace_context is None or not isinstance(vace_context, torch.Tensor):
                raise ValueError("vace_context is missing or not a tensor")
            else:
                total_frames = vace_context.shape[-3]
                control_frames = input_data.shape[-3]
                reference_frames = total_frames - control_frames - phantom_frames
                if reference_frames < 0:
                    raise ValueError("Reference frames count cannot be negative. Check vace_context and input data shapes.")
            
            # Store frame counts in conditioning for the patched function
            c["_vace_reference_frames"] = reference_frames
            c["_vace_phantom_frames"] = phantom_frames
            
            diffusion_model = model.get_model_object("diffusion_model")
            
            original_forward = diffusion_model.forward_orig
            diffusion_model.forward_orig = types.MethodType(_vaceph_forward_orig, diffusion_model)
            try:
                out = model_function(input_data, timestep, **c)
            finally:
                diffusion_model.forward_orig = original_forward
        else:
            # Use original behavior
            out = model_function(input_data, timestep, **c)
        
        return out
    
    model.set_model_unet_function_wrapper(unet_wrapper_function)
    
    logging.info("WanVaceAdvanced model patch applied")
    return model


def unwrap_vace_phantom_wan_model(model):
    """
    Remove the patch from a VaceWanModel.
    Note: Returns a fresh clone without the patch.
    """
    model_clone = model.clone()
    logging.info("WanVaceAdvanced model patch removed")
    return model_clone

def _vaceph_forward_orig(
    self,
    x,
    t,
    context,
    vace_context,
    vace_strength,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    """
    Custom forward implementation that applies separate strength values for all Vace reference and control frames.
    Also allows for separate Phantom frames compatibility.
    The Vace reference frames are at the beginning of the temporal dimension (dim 2).
    The Phantom frames are concatenated at the end of the temporal dimension (dim 2).
    This is adapted from the original VaceWanModel.forward_orig method.
    """
    
    # Extract frame count information for strength application
    phantom_frames = 0
    time_dim_concat = kwargs.get("time_dim_concat")
    if time_dim_concat is not None and isinstance(time_dim_concat, torch.Tensor):
        phantom_frames = time_dim_concat.shape[2]
    
    # embeddings
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    # Process vace_context
    orig_shape = list(vace_context.shape)
    vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
    c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
    vace_grid_sizes = c.shape[2:]  # Get Vace patch dimensions (t_patches, h_patches, w_patches)
    c = c.flatten(2).transpose(1, 2)
    
    # Store expected sequence length from x for comparison
    expected_seq_length = x.shape[1]
    
    # Extract reference frame count by analyzing the vace context structure
    # Reference frames have 32 channels (16 + 16 zeros), control frames have 32 channels (16 inactive + 16 reactive)
    # But we need to determine this from the strength list structure when we have it
    reference_frames = 0
    
    # Split into batch segments
    c = list(c.split(orig_shape[0], dim=0))
    
    # arguments
    x_orig = x

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    
    for i, block in enumerate(self.blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                return out
            out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
            x = out["img"]
        else:
            x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        ii = self.vace_layers_mapping.get(i, None)
        if ii is not None:
            for iii in range(len(c)):
                # Check and fix sequence length mismatch before passing to vace_blocks
                if c[iii].shape[1] < expected_seq_length:
                    # Calculate how many frames to add
                    seq_length_per_frame = grid_sizes[1] * grid_sizes[2]
                    extra_seq_length = expected_seq_length - c[iii].shape[1]
                    extra_frames = extra_seq_length // seq_length_per_frame
                    # Create a tensor of shape [1, 16, extra_frames, orig_shape[-2], orig_shape[-1]]
                    # Keep new tensors on CPU until added to the sequence
                    SCALE_FACTOR = 0.5  # Scale factor (what should this be?!)
                    solid_frames = torch.ones(1, 16, extra_frames, orig_shape[-2], orig_shape[-1], device="cpu", dtype=c[iii].dtype) * SCALE_FACTOR

                    from comfy.latent_formats import Wan21
                    processed_frames = Wan21().process_out(solid_frames)
                    processed_frames = self.patch_embedding(processed_frames.float()).to(processed_frames.dtype)
                    c_padding = processed_frames.flatten(2).transpose(1, 2)

                    # Concatenate along sequence dimension to extend c
                    c[iii] = torch.cat([c[iii], c_padding.to(c[iii].device)], dim=1)
                    
                    # if i == 0 and iii == 0:
                    #     logging.info(f"\nVace-Phantom compatibility: Extended Vace frames by adding {extra_frames} frames to match sequence length {expected_seq_length}.\n")

                    del processed_frames, c_padding, solid_frames

                elif c[iii].shape[1] > expected_seq_length:
                    # If c[iii] is longer than expected, truncate it
                    logging.warning(f"Truncating Vace frames for batch {iii} from {c[iii].shape[1]} to {expected_seq_length} frames.")
                    c[iii] = c[iii][:, :expected_seq_length, :]

                # Continue with original processing
                c_skip, c[iii] = self.vace_blocks[ii](
                    c[iii], x=x_orig, e=e0, freqs=freqs, 
                    context=context, context_img_len=context_img_len
                )
                
                # CUSTOM LOGIC: Apply separate reference strength if provided
                # iii is the batch index, vace_strength[iii] is the strength for this batch item
                if iii < len(vace_strength):
                    batch_strength = vace_strength[iii]
                else:
                    # Fallback to last strength if list is shorter
                    batch_strength = vace_strength[-1]
                
                # Extract frame counts from conditioning
                reference_frames = kwargs.get("_vace_reference_frames", 0)
                phantom_frames = kwargs.get("_vace_phantom_frames", 0)
                
                # Handle nested list (separate reference/control strengths) or single value
                if isinstance(batch_strength, list):
                    # We have separate strengths as a list
                    x = _apply_separate_strengths(x, c_skip, batch_strength, reference_frames, phantom_frames, vace_grid_sizes)
                else:
                    # Single strength value - original behavior
                    x += c_skip * batch_strength
                
            del c_skip

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x


def _apply_separate_strengths(x, c_skip, strength_list, reference_frames=0, phantom_frames=0, vace_grid_sizes=None):
    """
    Apply separate strength values for Vace frames with element-wise application.
    
    The sequence layout is: [reference frames] + [control frames] + [phantom frames]
    Each portion gets element-wise strength application based on the strength_list.
    
    Args:
        x: Main feature tensor [B, sequence_length, hidden_dim]
        c_skip: Vace skip connection [B, sequence_length, hidden_dim]
        strength_list: List of strengths matching the total frame count
        reference_frames: Number of reference frames (0 if no reference)
        phantom_frames: Number of phantom frames (0 if no phantom)
        vace_grid_sizes: Actual patch grid sizes (t_patches, h_patches, w_patches) from Vace embedding
    """
    # Get tensor dimensions
    batch_size, sequence_length, hidden_dim = c_skip.shape
    
    # Simple case: single strength for everything
    if len(strength_list) == 1 and not isinstance(strength_list[0], list):
        x += c_skip * strength_list[0]
        return x
    
    # Calculate control frames
    control_frames = len(strength_list[0]) - reference_frames - phantom_frames
    
    # Calculate patches per frame if grid sizes available
    if vace_grid_sizes is not None and len(vace_grid_sizes) >= 3:
        t_patches, h_patches, w_patches = vace_grid_sizes
        patches_per_frame = h_patches * w_patches
    else:
        # Fallback: assume equal distribution
        total_frames = reference_frames + control_frames + phantom_frames
        if total_frames > 0:
            patches_per_frame = sequence_length // total_frames
        else:
            patches_per_frame = sequence_length
            
    try:
        # Calculate sequence lengths for each portion
        reference_seq_length = reference_frames * patches_per_frame
        control_seq_length = control_frames * patches_per_frame
        phantom_seq_length = phantom_frames * patches_per_frame
        
        # Ensure we don't exceed the actual sequence length
        total_calculated = reference_seq_length + control_seq_length + phantom_seq_length
        if total_calculated > sequence_length:
            # Adjust phantom sequence length to fit
            phantom_seq_length = sequence_length - reference_seq_length - control_seq_length
            phantom_seq_length = max(0, phantom_seq_length)
        
        # Extract strength portions
        flat_strength_list = strength_list[0]  # Get the flat list for this batch
        reference_strengths = flat_strength_list[:reference_frames] if reference_frames > 0 else []
        control_strengths = flat_strength_list[reference_frames:reference_frames + control_frames]
        phantom_strengths = flat_strength_list[reference_frames + control_frames:] if phantom_frames > 0 else []
        
        # Split c_skip into portions
        current_pos = 0
        weighted_portions = []
        
        # Process reference frames
        if reference_seq_length > 0:
            ref_portion = c_skip[:, current_pos:current_pos + reference_seq_length, :]
            # Apply element-wise strengths to reference frames
            weighted_ref = torch.zeros_like(ref_portion)
            for i, strength in enumerate(reference_strengths):
                start_idx = i * patches_per_frame
                end_idx = min((i + 1) * patches_per_frame, reference_seq_length)
                weighted_ref[:, start_idx:end_idx, :] = ref_portion[:, start_idx:end_idx, :] * strength
            weighted_portions.append(weighted_ref)
            current_pos += reference_seq_length
        
        # Process control frames
        if control_seq_length > 0:
            ctrl_portion = c_skip[:, current_pos:current_pos + control_seq_length, :]
            # Apply element-wise strengths to control frames
            weighted_ctrl = torch.zeros_like(ctrl_portion)
            for i, strength in enumerate(control_strengths):
                start_idx = i * patches_per_frame
                end_idx = min((i + 1) * patches_per_frame, control_seq_length)
                weighted_ctrl[:, start_idx:end_idx, :] = ctrl_portion[:, start_idx:end_idx, :] * strength
            weighted_portions.append(weighted_ctrl)
            current_pos += control_seq_length
        
        # Process phantom frames (if any remaining sequence)
        if current_pos < sequence_length:
            phantom_portion = c_skip[:, current_pos:, :]
            if phantom_frames > 0 and phantom_strengths:
                # Apply element-wise strengths to phantom frames
                weighted_phantom = torch.zeros_like(phantom_portion)
                for i, strength in enumerate(phantom_strengths):
                    start_idx = i * patches_per_frame
                    end_idx = min((i + 1) * patches_per_frame, phantom_portion.shape[1])
                    if start_idx < phantom_portion.shape[1]:
                        weighted_phantom[:, start_idx:end_idx, :] = phantom_portion[:, start_idx:end_idx, :] * strength
                weighted_portions.append(weighted_phantom)
            else:
                # No phantom frames expected, use last control strength
                last_strength = control_strengths[-1] if control_strengths else 1.0
                weighted_portions.append(phantom_portion * last_strength)
        
        # Concatenate all weighted portions
        if weighted_portions:
            weighted_c_skip = torch.cat(weighted_portions, dim=1)
        else:
            # Fallback
            weighted_c_skip = c_skip * strength_list[0]
            
    except Exception as e:
        # Fallback to uniform application
        logging.warning(f"Failed to apply element-wise strengths, using uniform: {e}")
        flat_strength_list = strength_list[0] if isinstance(strength_list[0], list) else strength_list
        avg_strength = sum(flat_strength_list) / len(flat_strength_list)
        weighted_c_skip = c_skip * avg_strength
    
    # Add to main features
    x += weighted_c_skip
    
    return x

"""
HuMo I2V patching for WAN21_HuMo models.
Wraps zero-initialization in concat_latent_image check to enable I2V support.
"""

def is_wan21_humo_model(model):
    try:
        return isinstance(model.model, comfy.model_base.WAN21_HuMo)
    except Exception as e:
        logging.warning(f"Error checking model type: {e}")
        return False


def patch_humo_i2v_support(model):
    if not is_wan21_humo_model(model):
        wan_print(f"Warning: Model type is {type(model.model).__name__}, not WAN21_HuMo")
        raise ValueError("This patch is only compatible with WAN21_HuMo models")

    model_obj = model.model

    if hasattr(model_obj, '_humo_i2v_patched'):
        wan_print("Model already has HuMo I2V patch applied, skipping")
        return model

    def patched_extra_conds(self, **kwargs):
        out = super(type(self), self).extra_conds(**kwargs)
        noise = kwargs.get("noise", None)
        audio_embed = kwargs.get("audio_embed", None)

        if audio_embed is not None:
            out['audio_embed'] = comfy.conds.CONDRegular(audio_embed)

        concat_latent_image = kwargs.get("concat_latent_image", None)

        if "c_concat" not in out:
            reference_latents = kwargs.get("reference_latents", None)
            if reference_latents is not None:
                out['reference_latent'] = comfy.conds.CONDRegular(self.process_latent_in(reference_latents[-1]))
        else:
            if concat_latent_image is None:
                noise_shape = list(noise.shape)
                noise_shape[1] += 4
                concat_latent = torch.zeros(noise_shape, device=noise.device, dtype=noise.dtype)
                zero_vae_values_first = torch.tensor([0.8660, -0.4326, -0.0017, -0.4884, -0.5283, 0.9207, -0.9896, 0.4433, -0.5543, -0.0113, 0.5753, -0.6000, -0.8346, -0.3497, -0.1926, -0.6938]).view(1, 16, 1, 1, 1)
                zero_vae_values_second = torch.tensor([1.0869, -1.2370, 0.0206, -0.4357, -0.6411, 2.0307, -1.5972, 1.2659, -0.8595, -0.4654, 0.9638, -1.6330, -1.4310, -0.1098, -0.3856, -1.4583]).view(1, 16, 1, 1, 1)
                zero_vae_values = torch.tensor([0.8642, -1.8583, 0.1577, 0.1350, -0.3641, 2.5863, -1.9670, 1.6065, -1.0475, -0.8678, 1.1734, -1.8138, -1.5933, -0.7721, -0.3289, -1.3745]).view(1, 16, 1, 1, 1)
                concat_latent[:, 4:] = zero_vae_values
                concat_latent[:, 4:, :1] = zero_vae_values_first
                concat_latent[:, 4:, 1:2] = zero_vae_values_second
                out['c_concat'] = comfy.conds.CONDNoiseShape(concat_latent)

            reference_latents = kwargs.get("reference_latents", None)
            if reference_latents is not None:
                ref_latent = self.process_latent_in(reference_latents[-1])
                ref_latent_shape = list(ref_latent.shape)
                ref_latent_shape[1] += 4 + ref_latent_shape[1]
                ref_latent_full = torch.zeros(ref_latent_shape, device=ref_latent.device, dtype=ref_latent.dtype)
                ref_latent_full[:, 20:] = ref_latent
                ref_latent_full[:, 16:20] = 1.0
                out['reference_latent'] = comfy.conds.CONDRegular(ref_latent_full)

        return out

    model_obj.extra_conds = types.MethodType(patched_extra_conds, model_obj)
    model_obj._humo_i2v_patched = True

    wan_print("HuMo I2V patch applied successfully")
    return model


def unpatch_humo_i2v_support(model):
    if not is_wan21_humo_model(model):
        return model

    if not hasattr(model.model, '_humo_i2v_patched'):
        return model

    delattr(model.model, '_humo_i2v_patched')
    wan_print("HuMo I2V patch removed")
    return model
