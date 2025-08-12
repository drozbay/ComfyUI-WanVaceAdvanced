import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats

from .vace_encoding import encode_vace_advanced

class WanVacePhantomSimple:
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
            "control_masks": ("MASK", ),
            "vace_reference": ("IMAGE", ),
            "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "vace_ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "phantom_images": ("IMAGE", ),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "neg_phant_img", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "WanVaceAdvanced"

    def encode(self, positive, negative, vae, width, height, length, batch_size,
               vace_strength=1.0, vace_ref_strength=1.0,
               control_video=None, control_masks=None, vace_reference=None, phantom_images=None):

        vace_strength2 = 1.0
        vace_ref2_strength = 1.0
        control_video2 = None
        control_masks2 = None
        vace_reference_2 = None

        return encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
                                    vace_strength=vace_strength, vace_strength2=vace_strength2,
                                    vace_ref_strength=vace_ref_strength, vace_ref2_strength=vace_ref2_strength,
                                    control_video=control_video, control_masks=control_masks,
                                    vace_reference=vace_reference, control_video2=control_video2,
                                    control_masks2=control_masks2, vace_reference_2=vace_reference_2,
                                    phantom_images=phantom_images)


class WanVacePhantomDual:
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
            "control_video2": ("IMAGE",),
            "control_masks": ("MASK", ),
            "control_masks2": ("MASK",),
            "vace_reference": ("IMAGE", ),
            "vace_reference_2": ("IMAGE", ),
            "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "vace_strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "vace_ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "vace_ref_strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "phantom_images": ("IMAGE", ),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "neg_phant_img", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "WanVaceAdvanced"

    def encode(self, positive, negative, vae, width, height, length, batch_size,
               vace_strength=1.0, vace_strength2=1.0, vace_ref_strength=1.0, vace_ref_strength2=1.0,
               control_video=None, control_masks=None, vace_reference=None,
               control_video2=None, control_masks2=None, vace_reference_2=None,
               phantom_images=None):

        return encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
                                    vace_strength=vace_strength, vace_strength2=vace_strength2,
                                    vace_ref_strength=vace_ref_strength, vace_ref2_strength=vace_ref_strength2,
                                    control_video=control_video, control_masks=control_masks,
                                    vace_reference=vace_reference, control_video2=control_video2,
                                    control_masks2=control_masks2, vace_reference_2=vace_reference_2,
                                    phantom_images=phantom_images)


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
                            "vace_ref_strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                            # Phantom inputs
                            "phantom_images": ("IMAGE", ),
                            "phantom_mask_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Vace mask value for the Phantom embed region."}),
                            "phantom_control_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Padded vace embedded latents value for the Phantom embed region."}),
                            "phantom_vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "Vace strength value for the Phantom embed region. (Read: *NOT* the strength of the Phantom embeds, just the strength of the Vace embedding applied to the Phantom region.)"})
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "neg_phant_img", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "WanVaceAdvanced"

    EXPERIMENTAL = True

    def encode(self, positive, negative, vae, width, height, length, batch_size,
               vace_strength=1.0, vace_strength2=1.0, vace_ref_strength=None, vace_ref_strength2=None,
               control_video=None, control_masks=None, vace_reference=None,
               control_video2=None, control_masks2=None, vace_reference_2=None,
               phantom_images=None, phantom_mask_value=1.0, phantom_control_value=0.0, phantom_vace_strength=1.0
               ):

        return encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
                                    vace_strength=vace_strength, vace_strength2=vace_strength2,
                                    vace_ref_strength=vace_ref_strength, vace_ref2_strength=vace_ref_strength2,
                                    control_video=control_video, control_masks=control_masks,
                                    vace_reference=vace_reference, control_video2=control_video2,
                                    control_masks2=control_masks2, vace_reference_2=vace_reference_2,
                                    phantom_images=phantom_images, phantom_mask_value=phantom_mask_value,
                                    phantom_control_value=phantom_control_value, phantom_vace_strength=phantom_vace_strength
                                    )


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
            
            # Process the reference latent to have 32 channels like in the original code
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
    "WanVacePhantomSimple": WanVacePhantomSimple,
    "WanVacePhantomDual": WanVacePhantomDual,
    "WanVacePhantomExperimental": WanVacePhantomExperimental,
    "WanVaceToVideoLatent": WanVaceToVideoLatent,
    "VaceAdvancedModelPatch": VaceAdvancedModelPatch,
    "VaceStrengthTester": VaceStrengthTester,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVacePhantomSimple": "WanVacePhantomSimple",
    "WanVacePhantomDual": "WanVacePhantomDual",
    "WanVacePhantomExperimental": "WanVacePhantomExperimental",
    "WanVaceToVideoLatent": "WanVaceToVideoLatent",
    "VaceAdvancedModelPatch": "VaceAdvancedModelPatch",
    "VaceStrengthTester": "VaceStrengthTester",
}