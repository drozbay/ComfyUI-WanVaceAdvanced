import torch
import comfy.utils
import comfy.latent_formats
import comfy.model_management
import node_helpers

from .utils import resize_with_edge_padding, debug_save_images

def _reconcile_vace_embedding_lengths(vace_frames_list, vace_mask_list, vace_strength_list):
    if not vace_frames_list or len(vace_frames_list) == 0:
        return vace_frames_list, vace_mask_list, vace_strength_list
    
    # Find the maximum frame count across all VACE operations
    max_frames = 0
    frame_counts = []
    for frames in vace_frames_list:
        if frames is not None:
            frame_count = frames.shape[2]  # [b, c, frames, h, w]
            frame_counts.append(frame_count)
            max_frames = max(max_frames, frame_count)
        else:
            frame_counts.append(0)

    # If all frames are the same length, no padding needed
    if all(count == max_frames for count in frame_counts):
        return vace_frames_list, vace_mask_list, vace_strength_list
    
    reconciled_frames = []
    reconciled_masks = []
    reconciled_strengths = []
    
    for i, (frames, mask, strengths) in enumerate(zip(vace_frames_list, vace_mask_list, vace_strength_list)):
        if frames is None:
            reconciled_frames.append(frames)
            reconciled_masks.append(mask)
            reconciled_strengths.append(strengths)
            continue
            
        current_frames = frame_counts[i]
        frames_to_pad = max_frames - current_frames
        
        if frames_to_pad > 0:
            # Pad frames tensor at the beginning (for reference frames)
            # Shape: [b, c, frames, h, w]
            padding_shape = list(frames.shape)
            padding_shape[2] = frames_to_pad
            
            # Check if this is a 32-channel tensor (inactive + reactive concatenated)
            if padding_shape[1] == 32:
                # Create padding for both inactive and reactive parts
                half_shape = padding_shape.copy()
                half_shape[1] = 16
                inactive_padding = comfy.latent_formats.Wan21().process_out(
                    torch.zeros(half_shape, device=frames.device, dtype=frames.dtype)
                )
                reactive_padding = comfy.latent_formats.Wan21().process_out(
                    torch.zeros(half_shape, device=frames.device, dtype=frames.dtype)
                )
                padding_encoded = torch.cat([inactive_padding, reactive_padding], dim=1)
            else:
                raise ValueError(f"Unexpected frame tensor shape: {frames.shape}. Expected 32 channels for VACE encoding.")
            
            padded_frames = torch.cat([padding_encoded, frames], dim=2)
            reconciled_frames.append(padded_frames)
            
            # Pad mask tensor
            if mask is not None:
                # Shape: [batch, channels, frames, h, w] 
                mask_padding_shape = list(mask.shape)
                mask_padding_shape[2] = frames_to_pad
                mask_padding = torch.zeros(mask_padding_shape, device=mask.device, dtype=mask.dtype)
                padded_mask = torch.cat([mask_padding, mask], dim=2)
                reconciled_masks.append(padded_mask)
            else:
                reconciled_masks.append(mask)

            # Pad strength lists with the first element of each list
            if strengths is not None and len(strengths) > 0:
                # strengths is a list of lists: [[batch][frame_strengths]]
                padded_strengths = []
                for batch_strengths in strengths:
                    if isinstance(batch_strengths, list):
                        # Add zeros at the beginning for reference frames
                        padded_batch = [batch_strengths[0]] * frames_to_pad + batch_strengths
                        padded_strengths.append(padded_batch)
                    else:
                        padded_strengths.append(batch_strengths)
                reconciled_strengths.append(padded_strengths)
            else:
                reconciled_strengths.append(strengths)
        else:
            reconciled_frames.append(frames)
            reconciled_masks.append(mask)
            reconciled_strengths.append(strengths)
    
    # # Validate that all lists match in size
    # if len(reconciled_frames) != len(reconciled_masks) or len(reconciled_frames) != len(reconciled_strengths):
    #     raise ValueError(f"List size mismatch: frames={len(reconciled_frames)}, masks={len(reconciled_masks)}, strengths={len(reconciled_strengths)}")
    
    # # Validate that all frame tensors have the same temporal dimension
    # for i, frames in enumerate(reconciled_frames):
    #     if frames is not None:
    #         if frames.shape[2] != max_frames:
    #             raise ValueError(f"Frame tensor {i} has {frames.shape[2]} frames, expected {max_frames}")
    
    # # Validate that all mask tensors have matching dimensions
    # for i, mask in enumerate(reconciled_masks):
    #     if mask is not None and reconciled_frames[i] is not None:
    #         if mask.shape[2] != reconciled_frames[i].shape[2]:
    #             raise ValueError(f"Mask tensor {i} temporal dimension {mask.shape[2]} doesn't match frames {reconciled_frames[i].shape[2]}")
    
    # # Validate strength list lengths
    # for i, strengths in enumerate(reconciled_strengths):
    #     if strengths is not None and len(strengths) > 0:
    #         for j, batch_strengths in enumerate(strengths):
    #             if isinstance(batch_strengths, list) and len(batch_strengths) != max_frames:
    #                 raise ValueError(f"Strength list {i} batch {j} has {len(batch_strengths)} entries, expected {max_frames}")
    
    return reconciled_frames, reconciled_masks, reconciled_strengths

def encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
        vace_strength_1=1.0, vace_strength_2=1.0, vace_ref_strength_1=None, vace_ref_strength_2=None,
        control_video_1=None, control_masks_1=None, vace_reference_1=None,
        control_video_2=None, control_masks_2=None, vace_reference_2=None,
        phantom_images=None, phantom_mask_value=1.0, phantom_control_value=0.0, phantom_vace_strength=0.0,
        wva_options=None
        ):
    
    def _encode_latent(pixels):
        if wva_options is not None and wva_options.use_tiled_vae:
            return vae.encode_tiled(pixels, tile_x=512, tile_y=512, overlap=64, tile_t=64, overlap_t=8)
        else:
            return vae.encode(pixels)
    
    def _create_vace_lists(_control_video, _control_masks, _reference_image, _vace_strength, _vace_ref_strength):

        if _control_video is not None:
            _control_video = _control_video.clone()
        if _control_masks is not None:
            _control_masks = _control_masks.clone()
        if _reference_image is not None:
            _reference_image = _reference_image.clone()

        if _vace_ref_strength is None:
            if isinstance(_vace_strength, list):
                _vace_ref_strength = _vace_strength[0]
            else:
                _vace_ref_strength = _vace_strength

        # Calculate the additional length needed for Phantom images in Vace embeds
        num_phantom_images = 0 if phantom_images is None else len(phantom_images)
        vace_length = length + num_phantom_images * 4

        vace_latent_length = ((vace_length - 1) // 4) + 1
        control_video_original_length = length if _control_video is None else _control_video.shape[0]
        if _control_video is not None:
            _control_video = comfy.utils.common_upscale(_control_video[:control_video_original_length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if _control_video.shape[0] < vace_length:
                _control_video = torch.nn.functional.pad(_control_video, (0, 0, 0, 0, 0, 0, 0, vace_length - _control_video.shape[0]), value=0.5)
        else:
            _control_video = torch.ones((vace_length, height, width, 3)) * 0.5

        if isinstance(_vace_strength, list):
            if len(_vace_strength) != vace_latent_length:
                raise ValueError(f"If vace_strength is a list, it must have length {vace_latent_length}, got {len(_vace_strength)}")
            
        vace_references_encoded = None
        if _reference_image is not None:
            vace_references_scaled = comfy.utils.common_upscale(_reference_image[:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if wva_options is not None and getattr(wva_options, 'debug_save_images', False):
                debug_save_images(vace_references_scaled, prefix="vace_reference")
            vace_references_encoded = _encode_latent(vace_references_scaled[:, :, :, :3])
            vace_references_encoded = torch.cat([vace_references_encoded, comfy.latent_formats.Wan21().process_out(torch.zeros_like(vace_references_encoded))], dim=1)
            num_ref_frames = vace_references_encoded.shape[2]  # Get the number of reference frames
            if not isinstance(_vace_strength, list):
                if _vace_ref_strength != _vace_strength:
                    # Repeat ref_strength for each reference frame, then add control strengths
                    inner_strength_list = [_vace_ref_strength] * num_ref_frames + [_vace_strength] * vace_latent_length
                    vace_strength_list = [inner_strength_list] * batch_size
                else:
                    # If vace_strength is a single value and equal to vace_ref_strength, expand it to a list
                    inner_strength_list = [_vace_strength] * (vace_latent_length + num_ref_frames)
                    vace_strength_list = [inner_strength_list] * batch_size
            else:
                # If vace_strength is already a list, we assume it has the correct length for control frames
                # Add ref_strength for each reference frame
                vace_strength_list = [[_vace_ref_strength] * num_ref_frames + _vace_strength] * batch_size
        else:
            # If no references are provided, expand the vace_strength to match the vace_latent_length
            if isinstance(_vace_strength, list):
                # If vace_strength is already a list, use it directly (should already be correct length)
                vace_strength_list = [_vace_strength] * batch_size
            else:
                # If vace_strength is a single value, expand it
                vace_strength_list = [[_vace_strength] * vace_latent_length] * batch_size

        if not isinstance(_vace_strength, list):
            # Only if a custom vace_strength list is NOT provided
            # If phantom_vace_strength is set, override the the phantom embed frame strengths at the end
            if phantom_vace_strength >= 0 and num_phantom_images > 0:
                for batch_idx in range(batch_size):
                    for i in range(1, num_phantom_images + 1):
                        vace_strength_list[batch_idx][-i] = phantom_vace_strength

        print_strength_debug_info(vace_strength_list, vace_references_encoded, num_phantom_images)

        control_masks_original_length = length if _control_masks is None else _control_masks.shape[0]
        if _control_masks is not None:
            masks_latent = _control_masks
            if masks_latent.ndim == 3:
                masks_latent = masks_latent.unsqueeze(1)
            masks_latent = comfy.utils.common_upscale(masks_latent[:control_masks_original_length], width, height, "bilinear", "center").movedim(1, -1)
            if masks_latent.shape[0] < vace_length:
                masks_latent = torch.nn.functional.pad(masks_latent, (0, 0, 0, 0, 0, 0, 0, vace_length - masks_latent.shape[0]), value=1.0)
        else:
            masks_latent = torch.ones((vace_length, height, width, 1))

        # Modify the phantom-overlapping portions if phantom images exist
        if num_phantom_images > 0:
            # modify control video values for phantom padding region
            _control_video[control_video_original_length:, :, :, :] = phantom_control_value
            # modify mask values for phantom padding region
            masks_latent[control_masks_original_length:, :, :, :] = phantom_mask_value

        _control_video = _control_video - 0.5
        inactive = (_control_video * (1 - masks_latent)) + 0.5
        reactive = (_control_video * masks_latent) + 0.5

        inactive = _encode_latent(inactive[:, :, :, :3])
        reactive = _encode_latent(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if vace_references_encoded is not None:
            control_video_latent = torch.cat((vace_references_encoded, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        masks_latent = masks_latent.view(vace_length, height_mask, vae_stride, width_mask, vae_stride)
        masks_latent = masks_latent.permute(2, 4, 0, 1, 3)
        masks_latent = masks_latent.reshape(vae_stride * vae_stride, vace_length, height_mask, width_mask)
        masks_latent = torch.nn.functional.interpolate(masks_latent.unsqueeze(0), size=(vace_latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if vace_references_encoded is not None:
            mask_pad = torch.zeros_like(masks_latent[:, :vace_references_encoded.shape[2], :, :])
            masks_latent = torch.cat((mask_pad, masks_latent), dim=1)
            vace_latent_length += vace_references_encoded.shape[2]
            trim_latent = vace_references_encoded.shape[2]

        masks_latent = masks_latent.unsqueeze(0)

        return control_video_latent, masks_latent, vace_strength_list, vace_references_encoded, trim_latent
    
    if phantom_images is not None:
        phantom_images = phantom_images.clone()

    num_phantom_images = 0 if phantom_images is None else len(phantom_images)

    # Always process first VACE context
    control_video_latent_1, masks_latent_1, vace_strength_list_1, vace_references_encoded_1, trim_latent_1 = _create_vace_lists(control_video_1, control_masks_1, vace_reference_1, vace_strength_1, vace_ref_strength_1)
    
    # Prepare lists for conditioning with first VACE context
    vace_frames_list = [control_video_latent_1]
    vace_mask_list = [masks_latent_1]
    vace_strength_list = [vace_strength_list_1]
    trim_latent = trim_latent_1
    
    # Initialize second VACE variables
    vace_references_encoded_2 = None
    # Only process second VACE context if needed
    if control_video_2 is not None or control_masks_2 is not None or vace_reference_2 is not None:
        control_video_latent_2, masks_latent_2, vace_strength_list_2, vace_references_encoded_2, trim_latent_2 = _create_vace_lists(control_video_2, control_masks_2, vace_reference_2, vace_strength_2, vace_ref_strength_2)
        
        # Add second VACE to lists
        vace_frames_list.append(control_video_latent_2)
        vace_mask_list.append(masks_latent_2)
        vace_strength_list.append(vace_strength_list_2)
        trim_latent = max(trim_latent_1, trim_latent_2)
        
        # Reconcile lengths to ensure all VACE operations have matching dimensions
        vace_frames_list, vace_mask_list, vace_strength_list = _reconcile_vace_embedding_lengths(
            vace_frames_list, vace_mask_list, vace_strength_list
        )

    positive = node_helpers.conditioning_set_values(positive, {
        "vace_frames": vace_frames_list,
        "vace_mask": vace_mask_list,
        "vace_strength": vace_strength_list},
        append=True)
    negative = node_helpers.conditioning_set_values(negative, {
        "vace_frames": vace_frames_list,
        "vace_mask": vace_mask_list,
        "vace_strength": vace_strength_list},
        append=True)

    ######################
    # execute WanPhantomSubjectToVideo logic
    phantom_length = length
    # Create the latent from WanPhantomSubjectToVideo (this will be our output latent)
    latent = torch.zeros([batch_size, 16, ((phantom_length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())

    # WanPhantomSubjectToVideo uses the negative from WanVace as input and creates two outputs
    neg_phant_img = negative  # This becomes negative_img_text (with zeros)

    if phantom_images is not None:
        # Check phantom resize mode from wva_options
        phantom_resize_mode = "center"
        if wva_options is not None:
            phantom_resize_mode = getattr(wva_options, 'phantom_resize_mode', "center")
        
        if phantom_resize_mode == "pad_edge":
            phantom_images_scaled = resize_with_edge_padding(phantom_images[:num_phantom_images], width, height)
        else: # phantom_resize_mode = "center"
            phantom_images_scaled = comfy.utils.common_upscale(phantom_images[:num_phantom_images].movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        if wva_options is not None and getattr(wva_options, 'debug_save_images', False):
            debug_save_images(phantom_images_scaled, prefix="phantom", seg_index=None)
            
        latent_images = []
        for i in phantom_images_scaled:
            latent_images += [_encode_latent(i.unsqueeze(0)[:, :, :, :3])]
        concat_latent_image = torch.cat(latent_images, dim=2)

        # Check if there's already a phantom embedding and warn if overwriting
        if positive and len(positive) > 0 and "time_dim_concat" in positive[0][1]:
            print("[WARNING] Overwriting existing phantom embeddings in positive conditioning")
        if negative and len(negative) > 0 and "time_dim_concat" in negative[0][1]:
            print("[WARNING] Overwriting existing phantom embeddings in negative conditioning")

        # Apply phantom logic: positive gets the phantom images (overwrites any existing)
        positive = node_helpers.conditioning_set_values(positive, {"time_dim_concat": concat_latent_image})
        # negative_text (cond2 in original) gets the phantom images (overwrites any existing)
        negative = node_helpers.conditioning_set_values(negative, {"time_dim_concat": concat_latent_image})
        # neg_phant_img (negative in original) gets zeros instead of phantom images
        neg_phant_img = node_helpers.conditioning_set_values(negative, {"time_dim_concat": comfy.latent_formats.Wan21().process_out(torch.zeros_like(concat_latent_image))})

    # Adjust latent size if reference image was provided (WanVaceToVideo logic)
    if vace_references_encoded_1 is not None or vace_references_encoded_2 is not None:
        # Prepend zeros to match the reference image dimensions
        # (Should this even be necessary? vace_references_encoded_1 and vace_references_encoded_2 should already be the same size)
        # (may be able to just use the trim_latent value here...)
        number_to_pad = max(vace_references_encoded_1.shape[2] if vace_references_encoded_1 is not None else 0,
                            vace_references_encoded_2.shape[2] if vace_references_encoded_2 is not None else 0)
        latent_pad = torch.zeros([batch_size, 16, number_to_pad, height // 8, width // 8], device=latent.device, dtype=latent.dtype)
        latent = torch.cat([latent_pad, latent], dim=2)

    out_latent = {}
    out_latent["samples"] = latent

    return (positive, negative, neg_phant_img, out_latent, trim_latent)

def replace_vace_context(
    positive, negative, vae,
    control_video=None, control_masks=None, reference_image=None,
    context_index=0, vace_strength=-1.0, vace_ref_strength=-1.0
):
    """
    Replace VACE embeddings at context_index with new encoded images.
    Only encodes what's provided; preserves existing embeddings for None inputs.
    Use negative strength values to preserve existing strengths.
    """
    import copy

    if control_video is not None and control_masks is None:
        raise ValueError("control_masks is required when control_video is provided")

    if (control_video is not None or reference_image is not None) and vae is None:
        raise ValueError("VAE is required when image inputs are provided")

    if not positive or len(positive) == 0:
        raise ValueError("Positive conditioning is empty")

    cond_dict = positive[0][1]
    if 'vace_frames' not in cond_dict:
        raise ValueError("Positive conditioning does not contain VACE embeddings (no 'vace_frames' key)")

    vace_frames_list = cond_dict['vace_frames']
    vace_mask_list = cond_dict['vace_mask']
    vace_strength_list = cond_dict['vace_strength']

    if context_index >= len(vace_frames_list):
        raise ValueError(f"context_index {context_index} is out of range. Only {len(vace_frames_list)} contexts available.")

    existing_frames = vace_frames_list[context_index]
    existing_mask = vace_mask_list[context_index]
    existing_strength = vace_strength_list[context_index]

    if existing_frames.ndim == 5:
        _, _, T_existing, H_latent, W_latent = existing_frames.shape
    else:
        T_existing, _, H_latent, W_latent = existing_frames.shape

    height = H_latent * 8
    width = W_latent * 8

    # Count reference frames (have zero mask) vs control frames (non-zero mask)
    num_ref_frames = 0
    if existing_mask.ndim == 5:
        mask_temporal = existing_mask.shape[2]
    else:
        mask_temporal = existing_mask.shape[1]

    for t in range(min(mask_temporal, T_existing)):
        if existing_mask.ndim == 5:
            frame_mask = existing_mask[0, :, t, :, :]
        else:
            frame_mask = existing_mask[:, t, :, :]

        if frame_mask.abs().max() < 0.01:
            num_ref_frames += 1
        else:
            break

    num_control_frames = T_existing - num_ref_frames

    def _encode_latent(pixels):
        return vae.encode(pixels)

    new_control_latent = None
    new_mask_latent = None
    new_ref_latent = None

    if control_video is not None:
        control_length = control_video.shape[0]
        expected_latent_frames = ((control_length - 1) // 4) + 1

        if expected_latent_frames != num_control_frames:
            raise ValueError(
                f"Control video frame count mismatch. New video has {control_length} frames "
                f"({expected_latent_frames} latent frames), but existing context has {num_control_frames} control latent frames. "
                f"Please ensure the control video matches the existing context dimensions."
            )

        _control_video = comfy.utils.common_upscale(
            control_video.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        _masks = control_masks.clone()
        if _masks.ndim == 3:
            _masks = _masks.unsqueeze(1)
        _masks = comfy.utils.common_upscale(_masks, width, height, "bilinear", "center").movedim(1, -1)

        if _control_video.shape[0] < control_length:
            _control_video = torch.nn.functional.pad(
                _control_video, (0, 0, 0, 0, 0, 0, 0, control_length - _control_video.shape[0]), value=0.5
            )
        if _masks.shape[0] < control_length:
            _masks = torch.nn.functional.pad(
                _masks, (0, 0, 0, 0, 0, 0, 0, control_length - _masks.shape[0]), value=1.0
            )

        # Split into inactive/reactive regions based on mask
        _control_video = _control_video - 0.5
        inactive = (_control_video * (1 - _masks)) + 0.5
        reactive = (_control_video * _masks) + 0.5

        inactive_encoded = _encode_latent(inactive[:, :, :, :3])
        reactive_encoded = _encode_latent(reactive[:, :, :, :3])
        new_control_latent = torch.cat((inactive_encoded, reactive_encoded), dim=1)

        # Convert mask to latent space
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        vace_length = control_length
        vace_latent_length = ((vace_length - 1) // 4) + 1

        masks_latent = _masks.view(vace_length, height_mask, vae_stride, width_mask, vae_stride)
        masks_latent = masks_latent.permute(2, 4, 0, 1, 3)
        masks_latent = masks_latent.reshape(vae_stride * vae_stride, vace_length, height_mask, width_mask)
        masks_latent = torch.nn.functional.interpolate(
            masks_latent.unsqueeze(0),
            size=(vace_latent_length, height_mask, width_mask),
            mode='nearest-exact'
        ).squeeze(0)
        new_mask_latent = masks_latent.unsqueeze(0)

    if reference_image is not None:
        ref_scaled = comfy.utils.common_upscale(
            reference_image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        ref_encoded = _encode_latent(ref_scaled[:, :, :, :3])

        # VACE format: 16 encoded channels + 16 zero channels
        new_ref_latent = torch.cat([
            ref_encoded,
            comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_encoded))
        ], dim=1)

        new_num_ref_frames = new_ref_latent.shape[2]

        if num_ref_frames > 0 and new_num_ref_frames != num_ref_frames:
            raise ValueError(
                f"Reference image frame count mismatch. New reference has {new_num_ref_frames} frames, "
                f"but existing context has {num_ref_frames} reference frames."
            )

    def _build_strength_list(new_strength, existing_strengths, count, default_fallback=1.0):
        """Build strength list. Negative values preserve existing strengths."""
        result = []

        if isinstance(new_strength, (list, tuple)):
            for i in range(count):
                if i < len(new_strength):
                    val = new_strength[i]
                    if val < 0:
                        if existing_strengths and i < len(existing_strengths):
                            result.append(existing_strengths[i])
                        else:
                            result.append(default_fallback)
                    else:
                        result.append(val)
                else:
                    last_val = new_strength[-1] if new_strength else default_fallback
                    if last_val < 0:
                        if existing_strengths and i < len(existing_strengths):
                            result.append(existing_strengths[i])
                        else:
                            result.append(default_fallback)
                    else:
                        result.append(last_val)
        else:
            if new_strength < 0:
                if existing_strengths:
                    result = list(existing_strengths[:count])
                    while len(result) < count:
                        result.append(result[-1] if result else default_fallback)
                else:
                    result = [default_fallback] * count
            else:
                result = [new_strength] * count

        return result

    existing_ref_strengths = []
    existing_control_strengths = []
    if isinstance(existing_strength, list) and len(existing_strength) > 0:
        full_strengths = existing_strength[0]
        existing_ref_strengths = full_strengths[:num_ref_frames] if num_ref_frames > 0 else []
        existing_control_strengths = full_strengths[num_ref_frames:] if num_ref_frames < len(full_strengths) else []

    if control_video is not None and reference_image is not None:
        # Replace both reference and control
        replacement_frames = torch.cat((new_ref_latent, new_control_latent), dim=2)
        mask_pad = torch.zeros_like(new_mask_latent[:, :, :new_ref_latent.shape[2], :, :])
        replacement_mask = torch.cat((mask_pad, new_mask_latent), dim=2)

        new_num_ref_frames = new_ref_latent.shape[2]
        num_control_latent_frames = new_control_latent.shape[2]
        batch_size = len(existing_strength) if isinstance(existing_strength, list) else 1

        ref_strengths = _build_strength_list(vace_ref_strength, existing_ref_strengths, new_num_ref_frames, default_fallback=0.5)
        ctrl_strengths = _build_strength_list(vace_strength, existing_control_strengths, num_control_latent_frames, default_fallback=1.0)
        replacement_strength = [ref_strengths + ctrl_strengths] * batch_size

    elif control_video is not None:
        # Replace only control, preserve existing reference
        if num_ref_frames > 0:
            if existing_frames.ndim == 5:
                existing_ref = existing_frames[:, :, :num_ref_frames, :, :]
            else:
                existing_ref = existing_frames[:num_ref_frames, :, :, :]
            replacement_frames = torch.cat((existing_ref, new_control_latent), dim=2)

            if existing_mask.ndim == 5:
                existing_ref_mask = existing_mask[:, :, :num_ref_frames, :, :]
            else:
                existing_ref_mask = existing_mask[:, :num_ref_frames, :, :]
            replacement_mask = torch.cat((existing_ref_mask, new_mask_latent), dim=2)
        else:
            replacement_frames = new_control_latent
            replacement_mask = new_mask_latent

        num_control_latent_frames = new_control_latent.shape[2]
        batch_size = len(existing_strength) if isinstance(existing_strength, list) else 1

        ctrl_strengths = _build_strength_list(vace_strength, existing_control_strengths, num_control_latent_frames, default_fallback=1.0)
        replacement_strength = [list(existing_ref_strengths) + ctrl_strengths] * batch_size

    elif reference_image is not None:
        # Replace only reference, preserve existing control
        if num_ref_frames > 0:
            if existing_frames.ndim == 5:
                existing_control = existing_frames[:, :, num_ref_frames:, :, :]
                existing_control_mask = existing_mask[:, :, num_ref_frames:, :, :]
            else:
                existing_control = existing_frames[num_ref_frames:, :, :, :]
                existing_control_mask = existing_mask[:, num_ref_frames:, :, :]
        else:
            existing_control = existing_frames
            existing_control_mask = existing_mask

        replacement_frames = torch.cat((new_ref_latent, existing_control), dim=2)

        new_num_ref_frames = new_ref_latent.shape[2]
        mask_pad = torch.zeros([1, 64, new_num_ref_frames, H_latent, W_latent],
                               device=existing_control_mask.device, dtype=existing_control_mask.dtype)
        replacement_mask = torch.cat((mask_pad, existing_control_mask), dim=2)

        batch_size = len(existing_strength) if isinstance(existing_strength, list) else 1

        ref_strengths = _build_strength_list(vace_ref_strength, existing_ref_strengths, new_num_ref_frames, default_fallback=0.5)
        replacement_strength = [ref_strengths + list(existing_control_strengths)] * batch_size
    else:
        return (positive, negative if negative is not None else positive)

    new_positive = copy.deepcopy(positive)
    new_positive[0][1]['vace_frames'][context_index] = replacement_frames
    new_positive[0][1]['vace_mask'][context_index] = replacement_mask
    new_positive[0][1]['vace_strength'][context_index] = replacement_strength

    if negative is not None:
        new_negative = copy.deepcopy(negative)
        new_negative[0][1]['vace_frames'][context_index] = replacement_frames
        new_negative[0][1]['vace_mask'][context_index] = replacement_mask
        new_negative[0][1]['vace_strength'][context_index] = replacement_strength
    else:
        new_negative = new_positive

    return (new_positive, new_negative)


def format_strength_list(s_list):
    """Format a strength list for debug output."""
    if not s_list or len(s_list) == 0:
        return "[]"

    formatted_list = [f"{x:.2f}" for x in s_list]

    if len(formatted_list) > 2 and len(set(formatted_list)) == 1:
        return f"[{formatted_list[0]} ... {formatted_list[-1]}]"
    else:
        return f"[{', '.join(formatted_list)}]"


def print_strength_debug_info(vace_strength_list, vace_references_encoded, num_phantom_images):
    """Print VACE strength distribution for debugging."""
    formatted_inner_list = format_strength_list(vace_strength_list[0])
    print(f"Using vace_strength as list: {formatted_inner_list} (length {len(vace_strength_list[0])})")

    vace_reference_length = vace_references_encoded.shape[2] if vace_references_encoded is not None else 0
    reference_strength_part = vace_strength_list[0][:vace_reference_length] if vace_references_encoded is not None else []
    control_strength_part = vace_strength_list[0][vace_reference_length:-(num_phantom_images)] if num_phantom_images > 0 else vace_strength_list[0][vace_reference_length:]
    phantom_strength_part = vace_strength_list[0][-(num_phantom_images):] if num_phantom_images > 0 else []

    print(f"Reference strength: {format_strength_list(reference_strength_part)} (length {len(reference_strength_part)})")
    print(f"Control strength: {format_strength_list(control_strength_part)} (length {len(control_strength_part)})")
    print(f"Phantom images strength: {format_strength_list(phantom_strength_part)} (length {len(phantom_strength_part)})")