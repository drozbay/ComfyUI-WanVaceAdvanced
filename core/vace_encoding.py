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

def format_strength_list(s_list):
    """
    Format a strength list for printing.
    """
    if not s_list:
        return "[]"
    if len(s_list) == 0:
        return "[]"
    
    # Format all numbers to two decimal places
    formatted_list = [f"{x:.2f}" for x in s_list]
    
    # If all elements are the same and there are more than 2, summarize
    if len(formatted_list) > 2 and len(set(formatted_list)) == 1:
        return f"[{formatted_list[0]} ... {formatted_list[-1]}]"
    else:
        return f"[{', '.join(formatted_list)}]"


def print_strength_debug_info(vace_strength_list, vace_references_encoded, num_phantom_images):
    """
    Print debug information about VACE strength distribution.
    """
    # Format the inner list for printing
    formatted_inner_list = format_strength_list(vace_strength_list[0])
    print(f"Using vace_strength as list: {formatted_inner_list} (length {len(vace_strength_list[0])})")
    
    # Print the reference, control, and phantom strength parts
    vace_reference_length = vace_references_encoded.shape[2] if vace_references_encoded is not None else 0
    reference_strength_part = vace_strength_list[0][:vace_reference_length] if vace_references_encoded is not None else []
    control_strength_part = vace_strength_list[0][vace_reference_length:-(num_phantom_images)] if num_phantom_images > 0 else vace_strength_list[0][vace_reference_length:]
    phantom_strength_part = vace_strength_list[0][-(num_phantom_images):] if num_phantom_images > 0 else []
    
    print(f"Reference strength: {format_strength_list(reference_strength_part)} (length {len(reference_strength_part)})")
    print(f"Control strength: {format_strength_list(control_strength_part)} (length {len(control_strength_part)})")
    print(f"Phantom images strength: {format_strength_list(phantom_strength_part)} (length {len(phantom_strength_part)})")