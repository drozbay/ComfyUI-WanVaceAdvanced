# ComfyUI-WanVaceAdvanced

Advanced VACE nodes for Wan video models in ComfyUI.

## Example Output

![Example Output](assets/WanVacePhantom_Example.gif)

*Example showing Phantom embeddings with VACE control - [View workflow](example_workflows/WanVaceAdvanced_Phantom_Ex_1.0.json)*

## Overview

This node pack provides advanced VACE functionality for Wan video generation, allowing fine-grained control over Vace frame strengths with reference images, control videos, and optional phantom embeddings. The V2 nodes offer the most complete feature set with automatic reference frame detection and model integration.

## Important: Model Patching Required

**The model must be patched to work with these nodes.** You have two options:

1. **Use the model input/output on V2 nodes** - The V2 nodes can patch the model automatically when you connect a model to their input
2. **Use VaceAdvancedModelPatch separately** - Patch the model once and use it with multiple nodes

The patching is safe - if you don't use WanVaceAdvanced nodes in your workflow, the patched model behaves exactly like the original. Heck you can put as many of them as you want in there if you want to be sure!

## Quick Start

Recommended for most users:
- **WanVacePhantomSimpleV2** - Single VACE context with all features
- **WanVacePhantomDualV2** - Dual VACE contexts for complex control
- **VaceAdvancedModelPatch** - Patch models when not using V2 node model inputs

## Main Nodes

### WanVacePhantomSimpleV2
The go-to node for most VACE operations. Provides a single VACE context with full phantom embedding support.

**Key Features:**
- Single VACE context with reference image support
- Control video and mask inputs
- Phantom image embedding
- Optional latent input with automatic reference frame detection
- Built-in model patching when model is connected

**Inputs:**
- `positive/negative` - Conditioning inputs
- `vae` - VAE model for encoding
- `width/height/length` - Video dimensions
- `control_video` - (Optional) Video for VACE control
- `control_masks` - (Optional) Masks for control regions  
- `vace_reference` - (Optional) Reference image for VACE
- `vace_strength` - Control strength (default: 1.0)
- `vace_ref_strength` - Reference strength (default: 1.0)
- `phantom_images` - (Optional) Images for phantom embedding
- `model` - (Optional) Model to patch (outputs patched model)
- `latent` - (Optional) Input latent with smart reference handling (will trim or add frames to latent to account for changes in reference frames)

### WanVacePhantomDualV2
Node supporting two independent VACE contexts for complex control scenarios.
- Dual VACE contexts that can be controlled independently
- Each context has its own reference, control video, and strength settings
- Useful for combining different types of control (e.g., pose + depth)
- All features from SimpleV2

### VaceAdvancedModelPatch
Patches a model to enable advanced VACE features like per-frame strength control. Does the same thing as passing a model through the model input and output lines of a V2 node.

### WanVacePhantomExperimental
Experimental node with additional parameters for fine-tuning phantom behavior.

**Using Strength Lists:**
With a patched model, you can provide a list of Floats to strength inputs for per-frame control:
```python
# Example: Set higher strength for initial frames (eg. for continuation video generation with overlap frames).
# This list will set the strength to 1.0 for 9 frames (1 + 2 * 4) and then 0.5 for the remaining 40 frames (10 * 4):
vace_strength = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
```
You can use rgthree's `Power Puter` node or Kijai's `String to Float List` node to send a strength list into any of the `strength` inputs.
Note: The per-frame aspect is in terms of *latent* frames. So each strength value represents 4 actual frames once decoded.

### VaceStrengthTester
Utility node for testing different strength configurations.

## Workflow Example

![Workflow Screenshot](assets/WanVaceAdvanced_Phantom_Ex_1.0.png)

**Workflow File**: [`example_workflows/WanVaceAdvanced_Phantom_Ex_1.0.json`](example_workflows/WanVaceAdvanced_Phantom_Ex_1.0.json)

This example demonstrates:
- Using WanVacePhantomSimpleV2 with Phantom embeddings
- Applying VACE control using the "Layout" control method
- Using a VACE reference to control the setting

## Chaining Nodes

VACE nodes can be chained together - the conditioning outputs from one node can be fed into another. When chaining:
- VACE contexts accumulate (each node adds its contexts to the conditioning)
- Phantom embeddings from the latest node take precedence (with a warning if overwriting)
- Reference frames in latents are automatically detected and handled

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/drozbay/ComfyUI-WanVaceAdvanced
   ```
2. Restart ComfyUI
3. The nodes will appear under the "WanVaceAdvanced" category

## Required Dependencies

### Node Packs
These node packs are commonly used alongside WanVaceAdvanced:
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) - WAN video model support and utilities
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) - Video loading and saving

## Recommended Models

### Core Models
Choose one of these model configurations:

**Option 1: Separate Phantom + VACE Module**
- **Phantom 14b**: [Phantom-Wan-14B_fp16](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Phantom-Wan-14B_fp16.safetensors)
- **VACE Module** (choose one):
  - bf16: [Wan2_1-VACE_module_14B_bf16.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1-VACE_module_14B_bf16.safetensors)
  - fp8: [Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors)

**Option 2: Pre-merged VACE+Phantom Model**
- **VACE+Phantom Merge**: [Wan2.1_VACE_Phantom](https://huggingface.co/Inner-Reflections/Wan2.1_VACE_Phantom/tree/main) by InnerReflections

### Optional LoRAs
- **2.2 A14B Low Noise LoRA**: [Wan22_A14B_T2V_lora_extract_r64.safetensors](https://huggingface.co/drozbay/Wan2.2_A14B_lora_extract/blob/main/Wan22_A14B_T2V_lora_extract_r64.safetensors) - LoRA extracted from Wan2.2 Low Noise model. Seems to improve motion and quality without capabilities of Phantom or Vace.
- **CausVid LoRA**: [Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors) - Allows for lower step count generations (8-12 steps) with CFG 2.0 - 3.0.

- For more information on using Vace + Phantom models together, see [InnerReflections article](https://civitai.com/articles/17908)

## Phantom Usage Tips

When using Phantom embeddings:
- **Always use CFG > 1.0** - Phantom requires CFG to work effectively. Values between 1.5-3.5 typically work best.
- **Combining Negatives** - Use ComfyUI's built-in `Conditioning Combine` node to merge the two negative conditioning outputs (negative and neg_phant_img) when working with Phantom.
- **Image Selection** - You do not have to remove the background for your subject, just make sure it is not overly complicated and that your subject is the focus of the iamge. You can use multiple images per subject... probably. Try different angles of the same subject. This is an area that needs more testing, so share your results!

For some of my testing results, see the [WanTests page](https://drozbay.github.io/WanTests/).

## Other nodes

The following nodes are also available:
- `WanVacePhantomSimple` - Original simple node (requires separate model patching)
- `WanVacePhantomDual` - Original dual node (requires separate model patching)
- `WanVacePhantomExperimental` - Original experimental node (includes a bit more Phantom controls that are probably not interesting)
- `WanVaceToVideoLatent` - Latent-based VACE processing (probably not a good idea to begin with but if you want to experiment...)

- Implementation by ablejones (drozbay)