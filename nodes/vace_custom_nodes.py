import nodes
import node_helpers
import torch
import numpy as np
import scipy.ndimage
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
        return {"optional": {
            "model": ("MODEL", ),
            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            "vae": ("VAE", ),
            "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
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

    def encode(self, positive=None, negative=None, vae=None, width=832, height=480, length=81, batch_size=1,
               model=None, latent_in=None,
               vace_strength=1.0, vace_ref_strength=1.0,
               control_video=None, control_masks=None, vace_reference=None, phantom_images=None):

        if positive is None:
            raise ValueError("Positive conditioning is required. Please connect a conditioning input.")

        if negative is None:
            negative = positive

        if vae is None:
            raise ValueError("VAE is required. Please connect a VAE.")

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
        return {"optional": {
            "model": ("MODEL", ),
            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            "vae": ("VAE", ),
            "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
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

    def encode(self, positive=None, negative=None, vae=None, width=832, height=480, length=81, batch_size=1,
                model=None, latent_in=None,
                vace_strength_1=1.0, vace_strength_2=1.0, vace_ref_strength_1=1.0, vace_ref_strength_2=1.0,
                control_video_1=None, control_masks_1=None, vace_reference_1=None,
                control_video_2=None, control_masks_2=None, vace_reference_2=None,
                phantom_images=None):

        if positive is None:
            raise ValueError("Positive conditioning is required. Please connect a conditioning input.")

        if negative is None:
            negative = positive

        if vae is None:
            raise ValueError("VAE is required. Please connect a VAE.")

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
        return {"optional": {
                            "model": ("MODEL", ),
                            "positive": ("CONDITIONING", ),
                            "negative": ("CONDITIONING", ),
                            "vae": ("VAE", ),
                            "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                            "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                            "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
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

    def encode(self, positive=None, negative=None, vae=None, width=832, height=480, length=81, batch_size=1,
               model=None, latent_in=None,
               vace_strength_1=1.0, vace_strength_2=1.0, vace_ref_strength_1=None, vace_ref_strength_2=None,
               control_video_1=None, control_masks_1=None, vace_reference_1=None,
               control_video_2=None, control_masks_2=None, vace_reference_2=None,
               phantom_images=None, phantom_mask_value=1.0, phantom_control_value=0.0, phantom_vace_strength=1.0,
               wva_options=None,
               ):

        if positive is None:
            raise ValueError("Positive conditioning is required. Please connect a conditioning input.")

        if negative is None:
            negative = positive

        if vae is None:
            raise ValueError("VAE is required. Please connect a VAE.")

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
                    
            # Check for noise_mask that needs adjustment
            noise_mask = latent_in.get("noise_mask")

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

                    if noise_mask is not None:
                        # Calculate how many mask frames to prepend (eg. for reference frames)
                        # mask is [T, 1, H, W], latent is [B, C, T_latent, H, W]
                        mask_temporal = noise_mask.shape[0]
                        frames_per_latent = mask_temporal / actual_latent_frames
                        mask_frames_to_add = max(1, int(round(frames_per_latent * our_ref_frames)))

                        mask_padding = torch.ones((mask_frames_to_add,) + noise_mask.shape[1:], device=noise_mask.device, dtype=noise_mask.dtype)
                        noise_mask = torch.cat([mask_padding, noise_mask], dim=0)

                        processed_latent = dict(processed_latent)
                        processed_latent["noise_mask"] = noise_mask
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

# Based on StringToFloatList from ComfyUI-KJNodes (https://github.com/kijai/ComfyUI-KJNodes/blob/e81f33508b0821ea2f53f4f46a833fa6215626bd/nodes/nodes.py#L1021)
class StringToFloatListRanged:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "string" :("STRING", {"default": "1, 2, 3#2, 5.5", "multiline": True}),
                    }
                }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    CATEGORY = "WanVaceAdvanced"
    FUNCTION = "createlist"
    DESCRIPTION = "Converts a comma-separated string to a float list. Supports repeat notation using '#' (e.g., '3#5' repeats 3.0 five times). Mix regular numbers and repeats: '1, 2.5#3, 7' becomes [1.0, 2.5, 2.5, 2.5, 7.0]."

    def createlist(self, string):
        import re

        result = []
        items = [x.strip() for x in string.split(',')]
            
        # Pattern for number#count notation
        repeat_pattern = r'^(-?\d+(?:\.\d+)?)#(\d+)$'

        for item in items:
            match = re.match(repeat_pattern, item)
            if match:
                # Repeat notation: number#count
                number = float(match.group(1))
                count = int(match.group(2))
                result.extend([number] * count)
            else:
                # Regular number
                result.append(float(item))
            
        return (result,)


class WanNoiseMaskToLatentSpace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "spatial_method": (["max", "min", "mean", "area", "nearest"], {"default": "max",
                    "tooltip": "Spatial reduction: max=preserve any coverage, min=require full coverage, mean=average, area=area-based, nearest=nearest-exact"}),
                "temporal_method": (["max", "min", "mean", "first", "last"], {"default": "max",
                    "tooltip": "Temporal reduction: max/min/mean across 4-frame groups, or first/last frame only"}),
            },
            "optional": {
                "first_frame_special": ("BOOLEAN", {"default": True,
                    "tooltip": "True: first frame 1:1, rest grouped by 4. False: all frames grouped by 4 uniformly."}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8,
                    "tooltip": "Target latent width. 0 = auto (mask width / 8)"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8,
                    "tooltip": "Target latent height. 0 = auto (mask height / 8)"}),
                "expand_spatial": ("INT", {"default": 0, "min": -nodes.MAX_RESOLUTION, "max": nodes.MAX_RESOLUTION, "step": 1,
                    "tooltip": "Pixels to grow (+) or shrink (-) spatially before downscaling"}),
                "expand_temporal": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1,
                    "tooltip": "Frames to grow (+) or shrink (-) temporally before downscaling"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "convert"
    CATEGORY = "WanVaceAdvanced"

    def _apply_temporal_reduction(self, frames, method):
        if method == "max":
            return frames.max(dim=2)[0]
        elif method == "min":
            return frames.min(dim=2)[0]
        elif method == "mean":
            return frames.mean(dim=2)
        elif method == "first":
            return frames[:, :, 0, :, :]
        elif method == "last":
            return frames[:, :, -1, :, :]
        raise ValueError(f"Unknown temporal method: {method}")

    def _expand_temporal(self, mask_np, expand_temporal):
        if expand_temporal == 0:
            return mask_np
        result = mask_np.copy()
        footprint = np.ones((3, 1, 1))
        for _ in range(abs(expand_temporal)):
            if expand_temporal > 0:
                result = scipy.ndimage.grey_dilation(result, footprint=footprint, mode='nearest')
            else:
                result = scipy.ndimage.grey_erosion(result, footprint=footprint, mode='nearest')
        return result

    def _expand_spatial(self, mask_np, expand_spatial):
        if expand_spatial == 0:
            return mask_np
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        result = []
        for frame_idx in range(mask_np.shape[0]):
            frame = mask_np[frame_idx]
            for _ in range(abs(expand_spatial)):
                if expand_spatial < 0:
                    frame = scipy.ndimage.grey_erosion(frame, footprint=kernel)
                else:
                    frame = scipy.ndimage.grey_dilation(frame, footprint=kernel)
            result.append(frame)
        return np.stack(result, axis=0)

    def convert(self, mask, spatial_method, temporal_method, first_frame_special=True, target_width=0, target_height=0, expand_spatial=0, expand_temporal=0):
        import torch.nn.functional as F

        # Apply expansion/shrinking first
        if expand_temporal != 0 or expand_spatial != 0:
            if mask.ndim == 2:
                mask_for_expand = mask.unsqueeze(0)
            elif mask.ndim == 4:
                mask_for_expand = mask.squeeze(1) if mask.shape[1] == 1 else mask.squeeze(0)
            else:
                mask_for_expand = mask
            mask_np = mask_for_expand.cpu().numpy()
            if expand_temporal != 0:
                mask_np = self._expand_temporal(mask_np, expand_temporal)
            if expand_spatial != 0:
                mask_np = self._expand_spatial(mask_np, expand_spatial)
            mask = torch.from_numpy(mask_np).float()

        # Normalize to [B, T, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 4:
            raise ValueError(f"Unsupported mask dimension: {mask.ndim}")

        B, T, H, W = mask.shape
        latent_h = target_height if target_height > 0 else H // 8
        latent_w = target_width if target_width > 0 else W // 8

        if first_frame_special:
            latent_t = ((T - 1) // 4) + 1 if T > 1 else 1
        else:
            latent_t = (T + 3) // 4

        # Spatial reduction
        mask_spatial = mask.view(B * T, 1, H, W)
        kernel_h, kernel_w = H // latent_h, W // latent_w

        if spatial_method == "max":
            if kernel_h > 0 and kernel_w > 0:
                mask_spatial = F.max_pool2d(mask_spatial, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
            if mask_spatial.shape[-2] != latent_h or mask_spatial.shape[-1] != latent_w:
                mask_spatial = F.interpolate(mask_spatial, size=(latent_h, latent_w), mode='nearest')
        elif spatial_method == "min":
            if kernel_h > 0 and kernel_w > 0:
                mask_spatial = -F.max_pool2d(-mask_spatial, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
            if mask_spatial.shape[-2] != latent_h or mask_spatial.shape[-1] != latent_w:
                mask_spatial = F.interpolate(mask_spatial, size=(latent_h, latent_w), mode='nearest')
        elif spatial_method == "mean":
            if kernel_h > 0 and kernel_w > 0:
                mask_spatial = F.avg_pool2d(mask_spatial, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
            if mask_spatial.shape[-2] != latent_h or mask_spatial.shape[-1] != latent_w:
                mask_spatial = F.interpolate(mask_spatial, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        elif spatial_method == "area":
            mask_spatial = F.interpolate(mask_spatial, size=(latent_h, latent_w), mode='area')
        elif spatial_method == "nearest":
            mask_spatial = F.interpolate(mask_spatial, size=(latent_h, latent_w), mode='nearest-exact')

        mask_spatial = mask_spatial.view(B, T, latent_h, latent_w)

        # Temporal reduction
        if T == 1 or latent_t == 1:
            mask_out = mask_spatial
        elif first_frame_special:
            first_frame = mask_spatial[:, 0:1, :, :]
            remaining = mask_spatial[:, 1:, :, :]
            remaining_frames = remaining.shape[1]

            if remaining_frames == 0:
                mask_out = first_frame
            else:
                num_groups = (remaining_frames + 3) // 4
                if remaining_frames % 4 != 0:
                    pad_frames = 4 - (remaining_frames % 4)
                    padding = remaining[:, -1:, :, :].repeat(1, pad_frames, 1, 1)
                    remaining = torch.cat([remaining, padding], dim=1)
                remaining = remaining.view(B, num_groups, 4, latent_h, latent_w)
                remaining_reduced = self._apply_temporal_reduction(remaining, temporal_method)
                mask_out = torch.cat([first_frame, remaining_reduced], dim=1)
        else:
            total_frames = mask_spatial.shape[1]
            num_groups = (total_frames + 3) // 4
            if total_frames % 4 != 0:
                pad_frames = 4 - (total_frames % 4)
                padding = mask_spatial[:, -1:, :, :].repeat(1, pad_frames, 1, 1)
                mask_spatial = torch.cat([mask_spatial, padding], dim=1)
            mask_spatial = mask_spatial.view(B, num_groups, 4, latent_h, latent_w)
            mask_out = self._apply_temporal_reduction(mask_spatial, temporal_method)

        if mask_out.shape[1] > latent_t:
            mask_out = mask_out[:, :latent_t, :, :]

        if B == 1:
            mask_out = mask_out.squeeze(0)

        return (mask_out,)


class WanVaceWindowReferences:
    """
    Provides a batch of reference images to be distributed across context windows.
    Each context window receives the next reference image in round-robin order:
    window 0 → ref 0, window 1 → ref 1, etc., wrapping if more windows than references.

    This enables different reference images for different portions of a long video
    when using context windows for extended video generation.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_images": ("IMAGE",),  # Batch [N, H, W, C]
            },
            "optional": {
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16,
                    "tooltip": "Target width for reference images. Images will be resized to match."}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16,
                    "tooltip": "Target height for reference images. Images will be resized to match."}),
                "explicit_ref_mapping": ("STRING", {
                    "default": "",
                    "tooltip": "Explicit mapping of references to windows. Format: '0,1,2,1,0' maps ref[0] to window 0, "
                              "ref[1] to window 1, etc. If empty, uses round-robin. Extra windows repeat last entry."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "WanVaceAdvanced"

    def encode(self, positive, negative, vae, reference_images, width=832, height=480, explicit_ref_mapping=""):
        """
        Encode each reference image through VAE and store as a batch for per-window injection.

        The encoded references are stored in conditioning under 'window_reference_batch'.
        During context window processing, the callback will select the appropriate reference
        for each window using explicit_ref_mapping if provided, otherwise round-robin.

        Args:
            explicit_ref_mapping: Optional string like "0,1,2,1,0" for explicit window-to-ref mapping.
                                  If empty, uses round-robin (window_idx % num_refs).
        """
        import comfy.utils

        # Ensure we have at least one reference image
        if reference_images is None or len(reference_images) == 0:
            logging.warning("WanVaceWindowReferences: No reference images provided")
            return (positive, negative)

        num_refs = reference_images.shape[0]
        logging.info(f"WanVaceWindowReferences: Encoding {num_refs} reference image(s)")

        encoded_refs = []

        for i in range(num_refs):
            # Get single reference image [H, W, C]
            ref_image = reference_images[i:i+1]  # Keep batch dim [1, H, W, C]

            # Resize to target dimensions
            ref_scaled = comfy.utils.common_upscale(
                ref_image.movedim(-1, 1),  # [1, C, H, W]
                width, height,
                "bilinear", "center"
            ).movedim(1, -1)  # [1, H, W, C]

            # Encode through VAE (only RGB channels)
            ref_encoded = vae.encode(ref_scaled[:, :, :, :3])
            # ref_encoded shape: [1, 16, T_latent, H_latent, W_latent] where T_latent=1 for single image

            # For VACE injection, we need 32 channels: 16 for the reference + 16 zeros
            # This matches how vace_references_encoded is created in vace_encoding.py
            ref_with_zeros = torch.cat([
                ref_encoded,
                comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_encoded))
            ], dim=1)
            # ref_with_zeros shape: [1, 32, T_latent, H_latent, W_latent]

            encoded_refs.append(ref_with_zeros)

        logging.info(f"WanVaceWindowReferences: Encoded {len(encoded_refs)} references, shape: {encoded_refs[0].shape}")

        # Store the list of encoded references and optional mapping in conditioning
        # The callback will select the appropriate reference for each window
        cond_values = {"window_reference_batch": encoded_refs}
        if explicit_ref_mapping.strip():
            cond_values["window_reference_mapping"] = explicit_ref_mapping
            logging.info(f"WanVaceWindowReferences: Using explicit mapping: {explicit_ref_mapping}")

        positive = node_helpers.conditioning_set_values(positive, cond_values)
        negative = node_helpers.conditioning_set_values(negative, cond_values)

        return (positive, negative)


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
    "StringToFloatListRanged": StringToFloatListRanged,
    "WanMaskToLatentSpace": WanNoiseMaskToLatentSpace,
    "WanVaceWindowReferences": WanVaceWindowReferences,
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
    "StringToFloatListRanged": "StringToFloatListRanged",
    "WanNoiseMaskToLatentSpace": "WanNoiseMaskToLatentSpace",
    "WanVaceWindowReferences": "WanVaceRefsToContextWindows",
}