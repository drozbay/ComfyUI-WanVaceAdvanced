import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.latent_formats
import logging

from ..core.vace_encoding import encode_vace_advanced
from ..core.utils import WVAOptions, WVAPipe

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

    DEPRECATED = True

    def encode(self, positive, negative, vae, width, height, length, batch_size,
               vace_strength=1.0, vace_ref_strength=1.0,
               control_video=None, control_masks=None, vace_reference=None, phantom_images=None):

        vace_strength2 = 1.0
        vace_ref2_strength = 1.0
        control_video2 = None
        control_masks2 = None
        vace_reference_2 = None

        return encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
                                    vace_strength_1=vace_strength, vace_strength_2=vace_strength2,
                                    vace_ref_strength_1=vace_ref_strength, vace_ref_strength_2=vace_ref2_strength,
                                    control_video_1=control_video, control_masks_1=control_masks,
                                    vace_reference_1=vace_reference, control_video_2=control_video2,
                                    control_masks_2=control_masks2, vace_reference_2=vace_reference_2,
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

    DEPRECATED = True

    def encode(self, positive, negative, vae, width, height, length, batch_size,
               vace_strength=1.0, vace_strength2=1.0, vace_ref_strength=1.0, vace_ref_strength2=1.0,
               control_video=None, control_masks=None, vace_reference=None,
               control_video2=None, control_masks2=None, vace_reference_2=None,
               phantom_images=None):

        return encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
                                    vace_strength_1=vace_strength, vace_strength_2=vace_strength2,
                                    vace_ref_strength_1=vace_ref_strength, vace_ref_strength_2=vace_ref_strength2,
                                    control_video_1=control_video, control_masks_1=control_masks,
                                    vace_reference_1=vace_reference, control_video_2=control_video2,
                                    control_masks_2=control_masks2, vace_reference_2=vace_reference_2,
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
                            "phantom_vace_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "Vace strength value for the Phantom embed region. (Read: *NOT* the strength of the Phantom embeds, just the strength of the Vace embedding applied to the Phantom region.)"})
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "neg_phant_img", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "WanVaceAdvanced"

    EXPERIMENTAL = True
    DEPRECATED = True

    def encode(self, positive, negative, vae, width, height, length, batch_size,
               vace_strength=1.0, vace_strength2=1.0, vace_ref_strength=None, vace_ref_strength2=None,
               control_video=None, control_masks=None, vace_reference=None,
               control_video2=None, control_masks2=None, vace_reference_2=None,
               phantom_images=None, phantom_mask_value=1.0, phantom_control_value=0.0, phantom_vace_strength=1.0
               ):

        return encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
                                    vace_strength_1=vace_strength, vace_strength_2=vace_strength2,
                                    vace_ref_strength_1=vace_ref_strength, vace_ref_strength_2=vace_ref_strength2,
                                    control_video_1=control_video, control_masks_1=control_masks,
                                    vace_reference_1=vace_reference, control_video_2=control_video2,
                                    control_masks_2=control_masks2, vace_reference_2=vace_reference_2,
                                    phantom_images=phantom_images, phantom_mask_value=phantom_mask_value,
                                    phantom_control_value=phantom_control_value, phantom_vace_strength=phantom_vace_strength
                                    )


class WanVacePhantomSimpleV2:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            "vae": ("VAE", ),
            "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        },
        "optional": {
            "latent_in": ("LATENT", {"tooltip": "Optional latent input to continue from"}),
            "control_video": ("IMAGE", ),
            "control_masks": ("MASK", ),
            "vace_reference": ("IMAGE", ),
            "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "vace_ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "phantom_images": ("IMAGE", ),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("model", "positive", "negative", "neg_phant_img", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "WanVaceAdvanced"

    def encode(self, positive, negative, vae, width, height, length, batch_size,
               model = None, latent_in = None,
               vace_strength=1.0, vace_ref_strength=1.0,
               control_video=None, control_masks=None, vace_reference=None, phantom_images=None):

        vace_strength_2 = 1.0
        vace_ref_strength_2 = 1.0
        control_video_2 = None
        control_masks_2 = None
        vace_reference_2 = None

        # If a latent input is provided, ensure the node width/height match the latent's decoded pixel size.
        # Latent tensors are expected shape [..., C, T, H_latent, W_latent] -> pixel size = H_latent * vae_stride, W_latent * vae_stride
        vae_stride = 8
        try:
            if latent_in is not None and isinstance(latent_in, dict):
                latent_samples = latent_in.get("samples")
                if isinstance(latent_samples, torch.Tensor) and latent_samples.ndim >= 5:
                    latent_w = int(latent_samples.shape[-1])
                    latent_h = int(latent_samples.shape[-2])
                    new_width = latent_w * vae_stride
                    new_height = latent_h * vae_stride
                    if new_width != width or new_height != height:
                        logging.info(f"Latent input detected — adjusting node width/height from {width}x{height} to {new_width}x{new_height}")
                        width = new_width
                        height = new_height
        except Exception as e:
            logging.warning(f"Failed to auto-adjust width/height from latent input: {e}")

        result = encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
                                    vace_strength_1=vace_strength, vace_strength_2=vace_strength_2,
                                    vace_ref_strength_1=vace_ref_strength, vace_ref_strength_2=vace_ref_strength_2,
                                    control_video_1=control_video, control_masks_1=control_masks,
                                    vace_reference_1=vace_reference, control_video_2=control_video_2,
                                    control_masks_2=control_masks_2, vace_reference_2=vace_reference_2,
                                    phantom_images=phantom_images
                                    )

        # Unpack the result
        positive_out, negative_out, neg_phant_img_out, latent_out, trim_latent_out = result
        
        latent_out = _process_incoming_latent(latent_in, latent_out, height, width, length, vace_reference_list=[vace_reference, vace_reference_2])

        # Handle model patching if model was provided
        output_model = model
        if model is not None:
            from ..core.patches import wrap_vace_phantom_wan_model
            output_model = wrap_vace_phantom_wan_model(model.clone())
            
        return (output_model, positive_out, negative_out, neg_phant_img_out, latent_out, trim_latent_out)


class WanVacePhantomDualV2:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            "vae": ("VAE", ),
            "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
        },
        "optional": {
            # Latent
            "latent_in": ("LATENT", {"tooltip": "Optional latent input to continue from"}),
            # First Vace Control
            "control_video_1": ("IMAGE", ),
            "control_masks_1": ("MASK", ),
            "vace_reference_1": ("IMAGE", ),
            "vace_strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "vace_ref_strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            # Second Vace Control
            "control_video_2": ("IMAGE",),
            "control_masks_2": ("MASK",),
            "vace_reference_2": ("IMAGE", ),
            "vace_strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "vace_ref_strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            # Phantom input
            "phantom_images": ("IMAGE", ),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("model", "positive", "negative", "neg_phant_img", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "WanVaceAdvanced"

    def encode(self, positive, negative, vae, width, height, length, batch_size,
                model = None, latent_in = None,
                vace_strength_1=1.0, vace_strength_2=1.0, vace_ref_strength_1=1.0, vace_ref_strength_2=1.0,
                control_video_1=None, control_masks_1=None, vace_reference_1=None,
                control_video_2=None, control_masks_2=None, vace_reference_2=None,
                phantom_images=None):
        
        # If a latent input is provided, ensure the node width/height match the latent's decoded pixel size.
        # Latent tensors are expected shape [..., C, T, H_latent, W_latent] -> pixel size = H_latent * vae_stride, W_latent * vae_stride
        vae_stride = 8
        try:
            if latent_in is not None and isinstance(latent_in, dict):
                latent_samples = latent_in.get("samples")
                if isinstance(latent_samples, torch.Tensor) and latent_samples.ndim >= 5:
                    latent_w = int(latent_samples.shape[-1])
                    latent_h = int(latent_samples.shape[-2])
                    new_width = latent_w * vae_stride
                    new_height = latent_h * vae_stride
                    if new_width != width or new_height != height:
                        logging.info(f"Latent input detected — adjusting node width/height from {width}x{height} to {new_width}x{new_height}")
                        width = new_width
                        height = new_height
        except Exception as e:
            logging.warning(f"Failed to auto-adjust width/height from latent input: {e}")

        result = encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
                                    vace_strength_1=vace_strength_1, vace_strength_2=vace_strength_2,
                                    vace_ref_strength_1=vace_ref_strength_1, vace_ref_strength_2=vace_ref_strength_2,
                                    control_video_1=control_video_1, control_masks_1=control_masks_1,
                                    vace_reference_1=vace_reference_1, control_video_2=control_video_2,
                                    control_masks_2=control_masks_2, vace_reference_2=vace_reference_2,
                                    phantom_images=phantom_images
                                    )
        
        # Unpack the result
        positive_out, negative_out, neg_phant_img_out, latent_out, trim_latent_out = result
        
        latent_out = _process_incoming_latent(latent_in, latent_out, height, width, length, vace_reference_list=[vace_reference_1, vace_reference_2])

        # Handle model patching if model was provided
        output_model = model
        if model is not None:
            from ..core.patches import wrap_vace_phantom_wan_model
            output_model = wrap_vace_phantom_wan_model(model.clone())
            
        return (output_model, positive_out, negative_out, neg_phant_img_out, latent_out, trim_latent_out)


class WanVacePhantomExperimentalV2:
    """
    V2 version with optional latent input and smart reference frame handling.
    Supports state_info latents and automatic reference frame detection.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                            "model": ("MODEL", ),
                            "positive": ("CONDITIONING", ),
                            "negative": ("CONDITIONING", ),
                            "vae": ("VAE", ),
                            "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                            "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                            "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {
                            # Latent input
                            "latent_in": ("LATENT", {"tooltip": "Optional latent input to continue from"}),
                            # First VACE control
                            "control_video_1": ("IMAGE", ),
                            "control_masks_1": ("MASK", ),
                            "vace_reference_1": ("IMAGE", ),
                            "vace_strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                            "vace_ref_strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                            # Second VACE control
                            "control_video_2": ("IMAGE", {"tooltip": "Second control video input"}),
                            "control_masks_2": ("MASK", {"tooltip": "Second control masks input"}),
                            "vace_reference_2": ("IMAGE", ),
                            "vace_strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                            "vace_ref_strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                            # Phantom inputs
                            "phantom_images": ("IMAGE", ),
                            "phantom_mask_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Vace mask value for the Phantom embed region."}),
                            "phantom_control_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Padded vace embedded latents value for the Phantom embed region."}),
                            "phantom_vace_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "Vace strength value for the Phantom embed region. (Read: *NOT* the strength of the Phantom embeds, just the strength of the Vace embedding applied to the Phantom region.)"}),
                            # WVA Options
                            "wva_options": ("WVA_OPTIONS", {}),
                }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT",)
    RETURN_NAMES = ("model", "positive", "negative", "neg_phant_img", "latent", "trim_latent")
    FUNCTION = "encode"
    
    CATEGORY = "WanVaceAdvanced"
    EXPERIMENTAL = True

    def encode(self, positive, negative, vae, width, height, length, batch_size,
               model=None, latent_in=None,
               vace_strength_1=1.0, vace_strength_2=1.0, vace_ref_strength_1=None, vace_ref_strength_2=None,
               control_video_1=None, control_masks_1=None, vace_reference_1=None,
               control_video_2=None, control_masks_2=None, vace_reference_2=None,
               phantom_images=None, phantom_mask_value=1.0, phantom_control_value=0.0, phantom_vace_strength=1.0,
               wva_options=None,
               ):
        
        # If a latent input is provided, ensure the node width/height match the latent's decoded pixel size.
        # Latent tensors are expected shape [..., C, T, H_latent, W_latent] -> pixel size = H_latent * vae_stride, W_latent * vae_stride
        vae_stride = 8
        try:
            if latent_in is not None and isinstance(latent_in, dict):
                latent_samples = latent_in.get("samples")
                if isinstance(latent_samples, torch.Tensor) and latent_samples.ndim >= 5:
                    latent_w = int(latent_samples.shape[-1])
                    latent_h = int(latent_samples.shape[-2])
                    new_width = latent_w * vae_stride
                    new_height = latent_h * vae_stride
                    if new_width != width or new_height != height:
                        logging.info(f"Latent input detected — adjusting node width/height from {width}x{height} to {new_width}x{new_height}")
                        width = new_width
                        height = new_height
        except Exception as e:
            logging.warning(f"Failed to auto-adjust width/height from latent input: {e}")

        result = encode_vace_advanced(positive, negative, vae, width, height, length, batch_size,
                                      vace_strength_1=vace_strength_1, vace_strength_2=vace_strength_2,
                                      vace_ref_strength_1=vace_ref_strength_1, vace_ref_strength_2=vace_ref_strength_2,
                                      control_video_1=control_video_1, control_masks_1=control_masks_1,
                                      vace_reference_1=vace_reference_1, control_video_2=control_video_2,
                                      control_masks_2=control_masks_2, vace_reference_2=vace_reference_2,
                                      phantom_images=phantom_images, phantom_mask_value=phantom_mask_value,
                                      phantom_control_value=phantom_control_value, phantom_vace_strength=phantom_vace_strength,
                                      wva_options=wva_options
                                      )
        
        # Unpack the result
        positive_out, negative_out, neg_phant_img_out, latent_out, trim_latent_out = result
        
        latent_out = _process_incoming_latent(latent_in, latent_out, height, width, length, vace_reference_list=[vace_reference_1, vace_reference_2])
        
        # Handle model patching if model was provided
        output_model = model
        if model is not None:
            from ..core.patches import wrap_vace_phantom_wan_model
            output_model = wrap_vace_phantom_wan_model(model.clone())
    
        return (output_model, positive_out, negative_out, neg_phant_img_out, latent_out, trim_latent_out)


from ..core.patches import wrap_vace_phantom_wan_model, unwrap_vace_phantom_wan_model

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
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "WanVaceAdvanced"
    
    def patch_model(self, model,):
        m = model.clone()
        m = wrap_vace_phantom_wan_model(m)

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

def _process_incoming_latent(latent_in, latent_out, height, width, length, vace_reference_list):
    @staticmethod
    def _replace_reference_frames(tensor, num_frames):
        """Replace first N frames with zeros along temporal dimension (-3)"""
        if num_frames <= 0:
            return tensor
        
        tensor_copy = tensor.clone()
        if tensor.shape[-3] >= num_frames:
            # Replace the first N frames with zeros
            tensor_copy[..., :num_frames, :, :] = 0
        return tensor_copy
    
    @staticmethod  
    def _trim_reference_frames(tensor, num_frames):
        """Remove first N frames along temporal dimension (-3)"""
        if num_frames <= 0:
            return tensor
        if tensor.shape[-3] > num_frames:
            return tensor[..., num_frames:, :, :]
        # If we'd trim everything, return a single zero frame
        return torch.zeros_like(tensor[..., :1, :, :])
    
    @staticmethod
    def _add_reference_frames(tensor, num_frames):
        """Add N reference frames at the beginning along temporal dimension (-3)"""
        if num_frames <= 0:
            return tensor

        tensor_copy = tensor.clone()
        # Pad the beginning with zeros
        padding = torch.zeros_like(tensor_copy[..., :num_frames, :, :])
        tensor_copy = torch.cat([padding, tensor_copy], dim=-3)
        return tensor_copy

    # First determine if we will be adding reference frames, replacing existing reference frames, or trimming off reference frames:
    expected_latent_frames = ((length - 1) // 4) + 1
            
    # If any of the local vace_reference inputs are present, we need a reference frame
    our_ref_frames = 0
    for ref in vace_reference_list:
        if ref is not None:
            our_ref_frames = 1

    # Process incoming latent if provided
    processed_latent = None
    if latent_in is not None:
        latent_samples = latent_in.get("samples")
        if latent_samples is not None:
            # Validate latent dimensions
            expected_h = height // 8
            expected_w = width // 8
            actual_h = latent_samples.shape[3]
            actual_w = latent_samples.shape[4]
                    
            if actual_h != expected_h or actual_w != expected_w:
                raise ValueError(
                    f"Latent dimension mismatch: Expected {expected_h}x{expected_w} "
                    f"(from {height}x{width} pixels), but got {actual_h}x{actual_w}. "
                    f"Please ensure the latent was generated with the same resolution settings."
                )
                    
            actual_latent_frames = latent_samples.shape[2]

            detected_ref_frames = 0
            # Auto-detect if actual frames = expected + 1, assume 1 reference frame
            if actual_latent_frames == expected_latent_frames + 1:
                detected_ref_frames = 1
            elif actual_latent_frames == expected_latent_frames:
                detected_ref_frames = 0
            else:
                #something is wrong, length is not right
                raise ValueError(f"Unexpected latent input frame count: {actual_latent_frames}, expected: {expected_latent_frames}")

            # process the latent based on our decision
            ref_shape = latent_samples.shape
                    
            if our_ref_frames > 0:
                # We have new reference frames
                if detected_ref_frames > 0:
                    # Replace existing reference frames
                    processed_latent = _apply_to_latent_dict(
                        latent_in, ref_shape, _replace_reference_frames, detected_ref_frames
                    )
                else:
                    # Add new reference frames at the start
                    processed_latent = _apply_to_latent_dict(
                        latent_in, ref_shape, _add_reference_frames, our_ref_frames
                    )
            else:
                # We have no new reference frames
                if detected_ref_frames > 0:
                    # Trim existing reference frames
                    processed_latent = _apply_to_latent_dict(
                        latent_in, ref_shape, _trim_reference_frames, detected_ref_frames
                    )
                else:
                    # No references anywhere, use as-is
                    processed_latent = latent_in

    # If we processed an incoming latent, use it instead of the generated one
    if processed_latent is not None:
        # Adjust the processed latent to match expected dimensions if needed
        processed_samples = processed_latent.get("samples")
        expected_samples = latent_out.get("samples")
                
        if processed_samples is not None and expected_samples is not None:
            # Ensure the processed latent has the right dimensions
            if processed_samples.shape[2] < expected_samples.shape[2]:
                # Pad with zeros if too short
                pad_frames = expected_samples.shape[2] - processed_samples.shape[2]
                padding = comfy.latent_formats.Wan21().process_out(torch.zeros_like(processed_samples[..., :pad_frames, :, :]))
                logging.warning(f"Processed samples padded from {processed_samples.shape[2]} to {expected_samples.shape[2]}")
                processed_samples = torch.cat([processed_samples, padding], dim=2)
            elif processed_samples.shape[2] > expected_samples.shape[2]:
                # Truncate if too long
                logging.warning(f"Processed samples truncated from {processed_samples.shape[2]} to {expected_samples.shape[2]}")
                processed_samples = processed_samples[..., :expected_samples.shape[2], :, :]

            # Update the latent output
            latent_out = dict(processed_latent)
            latent_out["samples"] = processed_samples

    return latent_out

# Local latent handling (compatible with RES4LYF)
def _apply_to_latent_dict(obj, ref_shape, modify_func, *args, **kwargs):

    """
    Recursively traverse obj and apply modify_func to tensors whose last 5 dimensions
    match ref_shape's last 5 dimensions. Used for state_info compatibility.
    
    Args:
        obj: The object to traverse (dict, list, tuple, tensor, etc.)
        ref_shape: Reference tensor shape to match against
        modify_func: Function to apply to matching tensors. Should accept (tensor, *args, **kwargs)
        *args, **kwargs: Additional arguments passed to modify_func
    
    Returns:
        Modified structure with applicable tensors transformed
    """
    import torch

    if isinstance(obj, torch.Tensor):
        if obj.ndim >= 5:
            # Check if last 5 dims match reference
            obj_last5 = obj.shape[-5:]
            ref_last5 = ref_shape[-5:] if len(ref_shape) >= 5 else ref_shape
            if obj_last5 == ref_last5:
                return modify_func(obj, *args, **kwargs)
        return obj

    if isinstance(obj, dict):
        changed = False
        out = {}
        for k, v in obj.items():
            nv = _apply_to_latent_dict(v, ref_shape, modify_func, *args, **kwargs)
            changed |= (nv is not v)
            out[k] = nv
        return out if changed else obj

    if isinstance(obj, list):
        changed = False
        out = []
        for v in obj:
            nv = _apply_to_latent_dict(v, ref_shape, modify_func, *args, **kwargs)
            changed |= (nv is not v)
            out.append(nv)
        return out if changed else obj

    if isinstance(obj, tuple):
        changed = False
        out = []
        for v in obj:
            nv = _apply_to_latent_dict(v, ref_shape, modify_func, *args, **kwargs)
            changed |= (nv is not v)
            out.append(nv)
        return tuple(out) if changed else obj

    return obj


class WVAPipeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "control_masks": ("MASK",),
                "vace_reference": ("IMAGE",),
                "phantom_images": ("IMAGE",),
                "vace_ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "vae": ("VAE", {"tooltip": "VAE for encoding VACE data in DetailerHook"}),
            }
        }
    
    RETURN_TYPES = ("WVA_PIPE",)
    RETURN_NAMES = ("wva_pipe",)
    FUNCTION = "create_pipe"
    CATEGORY = "WanVaceAdvanced"
    
    def create_pipe(self, vace_strength=1.0, control_video=None, control_masks=None, 
                    vace_reference=None, phantom_images=None, vace_ref_strength=None, vae=None):
        
        pipe = WVAPipe(
            control_video_1=control_video,
            control_masks_1=control_masks,
            vace_reference_1=vace_reference,
            vace_strength_1=vace_strength,
            vace_ref_strength_1=vace_ref_strength,
            phantom_images=phantom_images,
            vae=vae
        )
        
        return (pipe,)


class WVAOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "use_tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for large images"}),
                "enable_debug_prints": ("BOOLEAN", {"default": True, "tooltip": "Enable detailed debugging output"}),
                "debug_save_phantom": ("BOOLEAN", {"default": False, "tooltip": "Save phantom images to temp folder for debugging"}),
                "phantom_resize_mode": (["center", "pad_edge"], {"default": "center", "tooltip": "How to resize phantom images: center=crop to fit, pad_edge=preserve aspect ratio with edge padding"}),
                "phantom_combined_negative": ("BOOLEAN", {"default": False, "tooltip": "Use Conditioning (Combine) for phantom negative conditioning."}),
            }
        }
    
    RETURN_TYPES = ("WVA_OPTIONS",)
    RETURN_NAMES = ("wva_options",)
    FUNCTION = "create_options"
    CATEGORY = "WanVaceAdvanced"
    
    def create_options(self, use_tiled_vae=False, phantom_resize_mode="center", 
                      enable_debug_prints=True, debug_save_phantom=False, phantom_combined_negative=False):
        
        options = WVAOptions(
            use_tiled_vae=use_tiled_vae,
            enable_debug_prints=enable_debug_prints,
            debug_save_images=debug_save_phantom,
            phantom_resize_mode=phantom_resize_mode,
            phantom_combined_negative=phantom_combined_negative
        )
        
        return (options,)


NODE_CLASS_MAPPINGS = {
    "WanVacePhantomSimple": WanVacePhantomSimple,
    "WanVacePhantomDual": WanVacePhantomDual,
    "WanVacePhantomExperimental": WanVacePhantomExperimental,
    "WanVacePhantomSimpleV2": WanVacePhantomSimpleV2,
    "WanVacePhantomDualV2": WanVacePhantomDualV2,
    "WanVacePhantomExperimentalV2": WanVacePhantomExperimentalV2,
    "WanVaceToVideoLatent": WanVaceToVideoLatent,
    "VaceAdvancedModelPatch": VaceAdvancedModelPatch,
    "VaceStrengthTester": VaceStrengthTester,
    "WVAPipeSimple": WVAPipeSimple,
    "WVAOptionsNode": WVAOptionsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVacePhantomSimple": "WanVacePhantomSimple",
    "WanVacePhantomDual": "WanVacePhantomDual",
    "WanVacePhantomExperimental": "WanVacePhantomExperimental",
    "WanVacePhantomSimpleV2": "WanVacePhantomSimpleV2",
    "WanVacePhantomDualV2": "WanVacePhantomDualV2",
    "WanVacePhantomExperimentalV2": "WanVacePhantomExperimentalV2",
    "WanVaceToVideoLatent": "WanVaceToVideoLatent",
    "VaceAdvancedModelPatch": "VaceAdvancedModelPatch",
    "VaceStrengthTester": "VaceStrengthTester",
    "WVAPipeSimple": "WVAPipeSimple",
    "WVAOptionsNode": "WVAOptions",
}