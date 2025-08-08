# ComfyUI-WanVaceAdvanced

Advanced/Experimental VACE nodes for WAN video models in ComfyUI.

## Overview

This stuff is pretty experimental and is probably not correct and you should probably not use it.
But if you are a glutton for punishment and want to experiment with me, then welcome.

This custom node pack provides advanced VACE functionality for WAN video models in ComfyUI.

### WanVacePhantomExperimental
- Combines VACE and Phantom functionality
- Supports dual VACE operations with separate strength controls
- Advanced phantom image integration
- Multi-reference image support

### WanVaceToVideoLatent  
- Latent-space VACE processing
- Direct latent input support
- Reference latent integration

### VaceWanModelPatcher
- Model patching for per-frame strength support
- Enable/disable phantom and reference frame processing

### VaceStrengthTester
- Utility for testing different strength combinations
- Custom strength list support
- Debugging and visualization aid

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory
2. Restart ComfyUI
3. The nodes will appear under the "WanVaceAdvanced" category

## Notes

- Simpler nodes will come later
- For a great Phantom VACE model merge to use with this node, please see [InnerReflections article](https://civitai.com/articles/17908)
- by ablejones (drozbay)
