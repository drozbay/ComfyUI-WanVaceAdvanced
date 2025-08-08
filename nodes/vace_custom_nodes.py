import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats


class WanVacePhantomExperimental:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                            "positive": ("CONDITIONING", ),
                            "negative": ("CONDITIONING", ),
                            "vae": ("VAE", ),
                            "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                            "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                            "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {
                            "control_video": ("IMAGE", ),
                            "control_video2": ("IMAGE", {"tooltip": "Second control video input"}),
                            "control_masks": ("MASK", ),
                            "control_masks2": ("MASK", {"tooltip": "Second control masks input"}),
                            "vace_reference": ("IMAGE", ),
                            "vace_reference_2": ("IMAGE", ),
                            "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                            "vace_strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                            "vace_ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                            "vace_ref2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                            # Phantom inputs
                            "phantom_images": ("IMAGE", ),
                            "phantom_mask_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Vace mask value for the Phantom embed region."}),
                            "phantom_control_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Padded vace embedded latents value for the Phantom embed region."}),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "neg_phant_img", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "WanVaceAdvanced"

    EXPERIMENTAL = True

    def encode(self, positive, negative, vae, width, height, length, batch_size,
               vace_strength=1.0, vace_strength2=1.0, vace_ref_strength=None, vace_ref2_strength=None,
               control_video=None, control_masks=None, vace_reference=None,
               control_video2=None, control_masks2=None, vace_reference_2=None,
               phantom_images=None, phantom_mask_value=0.0, phantom_control_value=0.0,
               ):

        if control_video is not None:
            control_video = control_video.clone()
        if control_masks is not None:
            control_masks = control_masks.clone()
        if vace_reference is not None:
            vace_reference = vace_reference.clone()
        if phantom_images is not None:
            phantom_images = phantom_images.clone()
        if control_video2 is not None:
            control_video2 = control_video2.clone()
        if control_masks2 is not None:
            control_masks2 = control_masks2.clone()
        if vace_reference_2 is not None:
            vace_reference_2 = vace_reference_2.clone()

        if vace_ref_strength is None:
            if isinstance(vace_strength, list):
                vace_ref_strength = vace_strength[0]
            else:
                vace_ref_strength = vace_strength
        if vace_ref2_strength is None:
            if isinstance(vace_strength2, list):
                vace_ref2_strength = vace_strength2[0]
            else:
                vace_ref2_strength = vace_strength2

        num_phantom_images = 0 if phantom_images is None else len(phantom_images)
        # Calculate the additional length needed for Phantom images in Vace embeds
        vace_length = length + num_phantom_images * 4

        ########################
        # execute WanVaceToVideo logic

        vace_latent_length = ((vace_length - 1) // 4) + 1
        original_vace_latent_length = vace_latent_length  # Save original before it gets modified
        control_video_original_length = 0 if control_video is None else control_video.shape[0]
        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:control_video_original_length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if control_video.shape[0] < vace_length:
                control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, vace_length - control_video.shape[0]), value=0.5)
        else:
            control_video = torch.ones((vace_length, height, width, 3)) * 0.5

        if isinstance(vace_strength, list):
            if len(vace_strength) != vace_latent_length:
                raise ValueError(f"If vace_strength is a list, it must have length {vace_latent_length}, got {len(vace_strength)}")
        
        vace_references_encoded = None
        if vace_reference is not None:
            vace_references_scaled = comfy.utils.common_upscale(vace_reference[:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            vace_references_encoded = vae.encode(vace_references_scaled[:, :, :, :3])
            vace_references_encoded = torch.cat([vace_references_encoded, comfy.latent_formats.Wan21().process_out(torch.zeros_like(vace_references_encoded))], dim=1)
            if not isinstance(vace_strength, list):
                if vace_ref_strength != vace_strength:
                    vace_strength_list = [[vace_ref_strength, vace_strength]] * batch_size
                else:
                    # If vace_strength is a single value and equal to vace_ref_strength, we need to expand it to a list
                    inner_strength_list = [vace_strength] * (vace_latent_length + vace_references_encoded.shape[2])
                    vace_strength_list = [inner_strength_list] * batch_size
            else:
                # If vace_strength is already a list, we assume it has the correct length
                vace_strength_list = [[vace_ref_strength] + vace_strength] * batch_size
        else:
            # If no references are provided, expand the vace_strength to match the vace_latent_length
            if isinstance(vace_strength, list):
                # If vace_strength is already a list, use it directly (should already be correct length)
                vace_strength_list = [vace_strength] * batch_size
            else:
                # If vace_strength is a single value, expand it
                vace_strength_list = [[vace_strength] * vace_latent_length] * batch_size

        def format_strength_list(s_list):
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

        control_masks_original_length = 0 if control_masks is None else control_masks.shape[0]
        if control_masks is not None:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:control_masks_original_length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < vace_length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, vace_length - mask.shape[0]), value=1.0)
        else:
            mask = torch.ones((vace_length, height, width, 1))
        
        # Modify the phantom-overlapping portions if phantom images exist
        if num_phantom_images > 0:
            # modify control video values for phantom padding region
            control_video[control_video_original_length:, :, :, :] = phantom_control_value
            # modify mask values for phantom padding region
            mask[control_masks_original_length:, :, :, :] = phantom_mask_value

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if vace_references_encoded is not None:
            control_video_latent = torch.cat((vace_references_encoded, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(vace_length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, vace_length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(vace_latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if vace_references_encoded is not None:
            mask_pad = torch.zeros_like(mask[:, :vace_references_encoded.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            vace_latent_length += vace_references_encoded.shape[2]
            trim_latent = vace_references_encoded.shape[2]

        mask = mask.unsqueeze(0)

        # Prepare lists for conditioning (first VACE is already processed)
        vace_frames_list = [control_video_latent]
        vace_mask_list = [mask]
        vace_strength_list_all = [vace_strength_list]

        # Process second VACE operation if provided
        if control_video2 is not None or control_masks2 is not None:
            print(f"Processing second VACE with strength {vace_strength2}")
            
            # Process second control video
            control_video2_original_length = 0 if control_video2 is None else control_video2.shape[0]
            if control_video2 is not None:
                control_video2 = comfy.utils.common_upscale(control_video2[:control_video2_original_length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                if control_video2.shape[0] < vace_length:
                    control_video2 = torch.nn.functional.pad(control_video2, (0, 0, 0, 0, 0, 0, 0, vace_length - control_video2.shape[0]), value=0.5)
            else:
                control_video2 = torch.ones((vace_length, height, width, 3)) * 0.5

            # Process second control masks
            control_masks2_original_length = 0 if control_masks2 is None else control_masks2.shape[0]
            if control_masks2 is not None:
                mask2 = control_masks2
                if mask2.ndim == 3:
                    mask2 = mask2.unsqueeze(1)
                mask2 = comfy.utils.common_upscale(mask2[:control_masks2_original_length], width, height, "bilinear", "center").movedim(1, -1)
                if mask2.shape[0] < vace_length:
                    mask2 = torch.nn.functional.pad(mask2, (0, 0, 0, 0, 0, 0, 0, vace_length - mask2.shape[0]), value=1.0)
            else:
                mask2 = torch.ones((vace_length, height, width, 1))
            
            # Modify phantom regions for second VACE
            if num_phantom_images > 0:
                control_video2[control_video2_original_length:, :, :, :] = phantom_control_value
                mask2[control_masks2_original_length:, :, :, :] = phantom_mask_value

            # Process second control video same as first
            control_video2 = control_video2 - 0.5
            inactive2 = (control_video2 * (1 - mask2)) + 0.5
            reactive2 = (control_video2 * mask2) + 0.5

            inactive2 = vae.encode(inactive2[:, :, :, :3])
            reactive2 = vae.encode(reactive2[:, :, :, :3])
            control_video_latent2 = torch.cat((inactive2, reactive2), dim=1)
            
            # Process references for second VACE (use vace_reference_2 if provided, otherwise reuse first references)
            vace_references_encoded2 = None
            if vace_reference_2 is not None:
                vace_reference_2_scaled = comfy.utils.common_upscale(vace_reference_2[:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                vace_references_encoded2 = vae.encode(vace_reference_2_scaled[:, :, :, :3])
                vace_references_encoded2 = torch.cat([vace_references_encoded2, comfy.latent_formats.Wan21().process_out(torch.zeros_like(vace_references_encoded2))], dim=1)
                
                # Validate that both VACE operations have the same number of reference frames for proper stacking
                if vace_references_encoded is not None and vace_references_encoded2.shape[2] != vace_references_encoded.shape[2]:
                    raise ValueError(f"First VACE has {vace_references_encoded.shape[2]} reference frames, but second VACE has {vace_references_encoded2.shape[2]} reference frames. They must match for proper tensor stacking.")
                
                control_video_latent2 = torch.cat((vace_references_encoded2, control_video_latent2), dim=2)
            elif vace_references_encoded is not None:
                # Reuse first references if no second references provided
                vace_references_encoded2 = vace_references_encoded
                control_video_latent2 = torch.cat((vace_references_encoded, control_video_latent2), dim=2)

            # Create strength list for second VACE using its specific references
            if isinstance(vace_strength2, list):
                if len(vace_strength2) != original_vace_latent_length:
                    raise ValueError(f"If vace_strength2 is a list, it must have length {original_vace_latent_length}, got {len(vace_strength2)}")
            
            if vace_references_encoded2 is not None:
                if not isinstance(vace_strength2, list):
                    if vace_ref2_strength != vace_strength2:
                        vace_strength_list2 = [[vace_ref2_strength, vace_strength2]] * batch_size
                    else:
                        inner_strength_list = [vace_strength2] * (original_vace_latent_length + vace_references_encoded2.shape[2])
                        vace_strength_list2 = [inner_strength_list] * batch_size
                else:
                    vace_strength_list2 = [[vace_ref2_strength] + vace_strength2] * batch_size
            else:
                if isinstance(vace_strength2, list):
                    vace_strength_list2 = [vace_strength2] * batch_size
                else:
                    vace_strength_list2 = [[vace_strength2] * original_vace_latent_length] * batch_size

            # Process mask same as first
            mask2 = mask2.view(vace_length, height_mask, vae_stride, width_mask, vae_stride)
            mask2 = mask2.permute(2, 4, 0, 1, 3)
            mask2 = mask2.reshape(vae_stride * vae_stride, vace_length, height_mask, width_mask)
            mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0), size=(original_vace_latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

            if vace_references_encoded2 is not None:
                mask2_pad = torch.zeros_like(mask2[:, :vace_references_encoded2.shape[2], :, :])
                mask2 = torch.cat((mask2_pad, mask2), dim=1)

            mask2 = mask2.unsqueeze(0)

            # Add to lists
            vace_frames_list.append(control_video_latent2)
            vace_mask_list.append(mask2)
            vace_strength_list_all.append(vace_strength_list2)

        positive = node_helpers.conditioning_set_values(positive, {
            "vace_frames": vace_frames_list,
            "vace_mask": vace_mask_list,
            "vace_strength": vace_strength_list_all},
            append=True)
        negative = node_helpers.conditioning_set_values(negative, {
            "vace_frames": vace_frames_list,
            "vace_mask": vace_mask_list,
            "vace_strength": vace_strength_list_all},
            append=True)

        ######################
        # execute WanPhantomSubjectToVideo logic
        phantom_length = length
        # Create the latent from WanPhantomSubjectToVideo (this will be our output latent)
        latent = torch.zeros([batch_size, 16, ((phantom_length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        
        # WanPhantomSubjectToVideo uses the negative from WanVace as input and creates two outputs
        neg_phant_img = negative  # This becomes negative_img_text (with zeros)
        
        if phantom_images is not None:
            phantom_images_scaled = comfy.utils.common_upscale(phantom_images[:num_phantom_images].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            latent_images = []
            for i in phantom_images_scaled:
                latent_images += [vae.encode(i.unsqueeze(0)[:, :, :, :3])]
            concat_latent_image = torch.cat(latent_images, dim=2)

            # Apply phantom logic: positive gets the phantom images
            positive = node_helpers.conditioning_set_values(positive, {"time_dim_concat": concat_latent_image})
            # negative_text (cond2 in original) gets the phantom images 
            negative = node_helpers.conditioning_set_values(negative, {"time_dim_concat": concat_latent_image})
            # neg_phant_img (negative in original) gets zeros instead of phantom images
            neg_phant_img = node_helpers.conditioning_set_values(negative, {"time_dim_concat": comfy.latent_formats.Wan21().process_out(torch.zeros_like(concat_latent_image))})

        # Adjust latent size if reference image was provided (WanVaceToVideo logic)
        if vace_references_encoded is not None:
            # Prepend zeros to match the reference image dimensions
            latent_pad = torch.zeros([batch_size, 16, vace_references_encoded.shape[2], height // 8, width // 8], device=latent.device, dtype=latent.dtype)
            latent = torch.cat([latent_pad, latent], dim=2)
        
        out_latent = {}
        out_latent["samples"] = latent
        
        return (positive, negative, neg_phant_img, out_latent, trim_latent)


from ..patches import wrap_vace_phantom_wan_model, unwrap_vace_phantom_wan_model

class WanVaceToVideoLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                },
                "optional": {"control_latent_input": ("LATENT", ),
                             "reference_latent": ("LATENT", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "WanVaceAdvanced"

    EXPERIMENTAL = True

    def encode(self, positive, negative, vae, width, height, length, batch_size, strength, control_latent_input=None, reference_latent=None):
        # Calculate dimensions for latent space
        vae_stride = 8
        height_latent = height // vae_stride
        width_latent = width // vae_stride
        latent_length = ((length - 1) // 4) + 1
        
        # Process control_latent_input if provided
        if control_latent_input is not None:
            # Extract the control frames from the latent input
            control_frames_latent = control_latent_input["samples"]
            
            # Get the number of latent frames in the input
            input_latent_length = control_frames_latent.shape[2]
            
            # TARGETED FIX: Create a full tensor of 0.5s in pixel space for the entire length
            # This will be encoded to create the proper inactive tensor
            full_pixel_tensor = torch.ones((length, height, width, 3), 
                                        device=comfy.model_management.intermediate_device()) * 0.5
            
            # Encode this full tensor to get the proper inactive latent
            inactive = vae.encode(full_pixel_tensor)
            
            # For reactive, we need to start with the same encoded 0.5s
            reactive = inactive.clone()
                        
            # Calculate the corresponding latent frames in the reactive tensor
            latent_frames_to_replace = min(input_latent_length, latent_length)
            
            # Replace the relevant portion of the reactive tensor with our input latent
            reactive[:, :, :latent_frames_to_replace] = control_frames_latent[:, :, :latent_frames_to_replace]
            
        else:
            # If no control latent provided, create blank latents
            # Create pixel-space frames with value 0.5
            blank_frames = torch.ones((length, height, width, 3), 
                                    device=comfy.model_management.intermediate_device()) * 0.5
            
            # Encode through VAE
            inactive = vae.encode(blank_frames)
            reactive = inactive.clone()
        
        # Combine inactive and reactive
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        
        # Create a mask of ones with the same temporal dimension as control_video_latent
        mask_temporal_length = control_video_latent.shape[2]
        mask = torch.ones([1, vae_stride * vae_stride, mask_temporal_length, height_latent, width_latent], 
                        device=control_video_latent.device)
        
        # Process reference latent if provided
        trim_latent = 0
        if reference_latent is not None:
            ref_s = reference_latent["samples"]
            
            # Process the reference latent to have 32 channels like in the original
            # This matches how the original node processes reference images
            ref_s = torch.cat([ref_s, comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_s))], dim=1)
            
            # Concatenate with control video latent along temporal dimension
            control_video_latent = torch.cat((ref_s, control_video_latent), dim=2)
            trim_latent = ref_s.shape[2]  # Set trim amount to reference frame count

        # If we have a reference latent, we need to add zeros to mask for those frames
        if reference_latent is not None:
            mask_pad = torch.zeros([1, vae_stride * vae_stride, trim_latent, height_latent, width_latent], 
                                device=control_video_latent.device)
            mask = torch.cat((mask_pad, mask), dim=2)
        
        # Set conditioning values
        positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
        
        # Create output latent
        output_latent_length = mask_temporal_length
        if reference_latent is not None:
            output_latent_length += trim_latent
            
        latent = torch.zeros([batch_size, 16, output_latent_length, height_latent, width_latent], 
                            device=comfy.model_management.intermediate_device())
        
        out_latent = {}
        out_latent["samples"] = latent
        
        return (positive, negative, out_latent, trim_latent)
    

class VaceAdvancedModelPatch:
    """
    Node to enable per-frame strength support for VaceWanModel.
    This patches the model to support separate strength values for reference and control frames.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "WanVaceAdvanced"
    
    def patch_model(self, model, enable):
        """
        Wrap or unwrap the VaceWanModel to enable/disable per-frame strength support.
        
        Args:
            model: The model to patch
            enable: Whether to enable or disable per-frame strength support
            
        Returns:
            The patched model
        """
        # Clone the model to avoid modifying the original
        m = model.clone()
        
        if enable:
            m = wrap_vace_phantom_wan_model(m)
        else:
            m = unwrap_vace_phantom_wan_model(m)

        return (m,)


class VaceStrengthTester:
    """
    A utility node to test different strength combinations for VACE models.
    This node helps visualize how different reference and control strengths affect the output.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "frame_count": ("INT", {"default": 20, "min": 1, "max": 1000}),
            },
            "optional": {
                "custom_strength_list": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "strength_info")
    FUNCTION = "create_test_conditioning"
    CATEGORY = "WanVaceAdvanced"
    
    def create_test_conditioning(self, positive, reference_strength, control_strength, frame_count, custom_strength_list=""):
        """
        Create conditioning with test strength values.
        
        Args:
            positive: Input conditioning
            reference_strength: Strength for reference frame
            control_strength: Strength for control frames
            frame_count: Number of frames
            custom_strength_list: Optional custom list of strength values (comma-separated)
            
        Returns:
            Modified conditioning and info string
        """
        
        # Parse custom strength list if provided
        if custom_strength_list.strip():
            try:
                strength_list = [float(x.strip()) for x in custom_strength_list.split(",")]
                # Pad or truncate to frame_count
                if len(strength_list) < frame_count:
                    strength_list.extend([strength_list[-1]] * (frame_count - len(strength_list)))
                else:
                    strength_list = strength_list[:frame_count]
                
                info = f"Using custom strength list: {strength_list[:10]}{'...' if len(strength_list) > 10 else ''}"
                
            except ValueError:
                # Fall back to reference + control pattern
                strength_list = [reference_strength] + [control_strength] * (frame_count - 1)
                info = f"Invalid custom list, using ref:{reference_strength}, ctrl:{control_strength} for {frame_count} frames"
        else:
            # Standard reference + control pattern
            strength_list = [reference_strength] + [control_strength] * (frame_count - 1)
            info = f"Using ref:{reference_strength}, ctrl:{control_strength} for {frame_count} frames"
        
        # Clone conditioning to avoid modifying original
        import copy
        new_positive = copy.deepcopy(positive)
        
        # Find and update vace_strength in conditioning
        for i, cond_item in enumerate(new_positive):
            if len(cond_item) > 1 and isinstance(cond_item[1], dict):
                if 'vace_strength' in cond_item[1]:
                    cond_item[1]['vace_strength'] = strength_list
                    cond_item[1]['vace_has_reference'] = True
                    info += f" | Applied to conditioning item {i}"
        
        return (new_positive, info)


NODE_CLASS_MAPPINGS = {
    "WanVacePhantomExperimental": WanVacePhantomExperimental,
    "WanVaceToVideoLatent": WanVaceToVideoLatent,
    "VaceAdvancedModelPatch": VaceAdvancedModelPatch,
    "VaceStrengthTester": VaceStrengthTester,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVacePhantomAdvanced": "WanVacePhantomAdvanced",
    "WanVaceToVideoLatent": "WanVaceToVideoLatent",
    "VaceAdvancedModelPatch": "VaceAdvancedModelPatch",
    "VaceStrengthTester": "VaceStrengthTester",
}