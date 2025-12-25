import logging
import torch
import comfy.patcher_extension
import comfy.context_windows

USE_RETAIN_INDEX_LIST_FOR_VACE_STRENGTH_LIST = True # for testing

def _parse_ref_mapping(mapping_str: str) -> list:
    """
    Parse mapping string like "0,1,2" into list of integers.
    Returns None if empty/invalid (fall back to default).
    """
    if not mapping_str or not mapping_str.strip():
        return None
    try:
        parts = [p.strip() for p in mapping_str.split(",") if p.strip()]
        indices = [int(p) for p in parts]
        if any(i < 0 for i in indices):
            logging.warning(f"[WanVaceAdvanced] Negative indices in ref mapping: {mapping_str}. Using default.")
            return None
        return indices if indices else None
    except ValueError as e:
        logging.warning(f"[WanVaceAdvanced] Invalid ref mapping '{mapping_str}': {e}. Using default.")
        return None


def _create_window_tracker_callback(context_handler):
    """
    Creates a callback that tracks the current window index.

    This callback is registered with EVALUATE_CONTEXT_WINDOWS which is called
    BEFORE each window is processed. It stores the window_idx on the context_handler
    so that RESIZE_COND_ITEM callbacks can access it for per-window operations.
    """

    def evaluate_callback(handler, model, x_in, conds, timestep, model_options,
                          window_idx, window, *args, **kwargs):
        context_handler._wva_current_window_idx = window_idx

    return evaluate_callback


def _create_vace_callback(context_handler):
    """
    Creates a callback that handles VACE conditioning resize for context windows.
    """
    retain_index_list = getattr(context_handler, 'cond_retain_index_list', [])
    dim = 2

    def vace_resize_cond_item_callback(cond_key, cond_value, window, x_in, device, new_cond_item):
        # Handle vace_context - temporal dim is 3: [B, N, 96, T, H, W]
        if cond_key == "vace_context":
            if not (hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor)):
                return None

            vace_cond = cond_value.cond
            if vace_cond.ndim <= 3:
                return None

            x_temporal = x_in.size(dim)
            vace_temporal = vace_cond.size(3)

            # Detect Phantom (time_dim_concat present AND vace_context extended)
            has_time_dim_concat = "time_dim_concat" in new_cond_item
            has_extended_vace = vace_temporal > x_temporal

            if has_time_dim_concat and has_extended_vace:
                # Phantom mode: split, slice main with retain_index_list, re-append phantom
                T_phantom = vace_temporal - x_temporal
                vace_main = vace_cond[:, :, :, :-T_phantom, :, :]
                vace_phantom = vace_cond[:, :, :, -T_phantom:, :, :].to(device)
                sliced_main = window.get_tensor(vace_main, device, dim=3, retain_index_list=retain_index_list)
                sliced_vace = torch.cat([sliced_main, vace_phantom], dim=3)

                # Apply per-window reference injection if available
                sliced_vace = _inject_window_reference(
                    sliced_vace, new_cond_item, context_handler, device
                )

                return cond_value._copy_with(sliced_vace)

            elif vace_temporal == x_temporal:
                # Standard VACE (no phantom)
                sliced_vace = window.get_tensor(vace_cond, device, dim=3, retain_index_list=retain_index_list)

                # Apply per-window reference injection if available
                sliced_vace = _inject_window_reference(
                    sliced_vace, new_cond_item, context_handler, device
                )

                return cond_value._copy_with(sliced_vace)

            return None

        # Handle vace_strength (CONDConstant wrapping list)
        if cond_key == "vace_strength":
            if not (hasattr(cond_value, "cond") and isinstance(cond_value.cond, list)):
                return None

            inner_list = cond_value.cond
            if len(inner_list) == 0:
                return None

            x_temporal = x_in.size(dim)

            # Flat list of strengths
            if isinstance(inner_list[0], (int, float)):
                if len(inner_list) == x_temporal:
                    sliced = [inner_list[i] for i in window.index_list]
                    # Apply retain_index_list to strengths as well
                    if USE_RETAIN_INDEX_LIST_FOR_VACE_STRENGTH_LIST: # for testing, not sure if this does anything...
                        for idx in retain_index_list:
                            if idx < len(inner_list) and idx < len(sliced):
                                sliced[idx] = inner_list[idx]
                                logging.debug(f"[VACE Strength] Retaining index {idx} value {inner_list[idx]}")
                    return cond_value._copy_with(sliced)
                return None

            # List of lists (per-frame strengths with reference/control/phantom)
            if isinstance(inner_list[0], list):
                sliced_list = []
                for sub_list in inner_list:
                    sub_sub_list = sub_list[0]
                    if len(sub_sub_list) > x_temporal:
                        # Phantom: slice main portion, preserve phantom at end
                        T_phantom = len(sub_sub_list) - x_temporal
                        main_strengths = sub_sub_list[:-T_phantom]
                        phantom_strengths = sub_sub_list[-T_phantom:]
                        sliced_main = [main_strengths[i] for i in window.index_list]
                        # Apply retain_index_list to main strengths
                        if USE_RETAIN_INDEX_LIST_FOR_VACE_STRENGTH_LIST: # for testing, not sure if this does anything...
                            for idx in retain_index_list:
                                if idx < len(main_strengths) and idx < len(sliced_main):
                                    sliced_main[idx] = main_strengths[idx]
                                    logging.debug(f"[VACE Strength] Retaining index {idx} value {main_strengths[idx]}")
                        sliced_list.append([sliced_main + phantom_strengths])
                    elif len(sub_sub_list) == x_temporal:
                        sliced = [sub_sub_list[i] for i in window.index_list]
                        # Apply retain_index_list
                        if USE_RETAIN_INDEX_LIST_FOR_VACE_STRENGTH_LIST: # for testing, not sure if this does anything...
                            for idx in retain_index_list:
                                if idx < len(sub_sub_list) and idx < len(sliced):
                                    sliced[idx] = sub_sub_list[idx]
                                    logging.debug(f"[VACE Strength] Retaining index {idx} value {sub_sub_list[idx]}")
                        sliced_list.append([sliced])
                    else:
                        sliced_list.append([sub_sub_list])
                return cond_value._copy_with(sliced_list)

            return None

        return None  # Not handled

    return vace_resize_cond_item_callback


def _inject_window_reference(sliced_vace, new_cond_item, context_handler, device):
    """
    Injects a per-window reference image into the sliced vace_context.
    Args:
        sliced_vace: The sliced vace_context tensor [B, num_ops, 96, T, H, W]
        new_cond_item: The model_conds dict containing window_reference_batch
        context_handler: The context handler with current window index
        device: Target device
    Returns:
        Modified sliced_vace
    """
    ref_batch_cond = new_cond_item.get("window_reference_batch", None)
    if ref_batch_cond is None:
        return sliced_vace

    # Extract batch from CONDConstant wrapper
    if hasattr(ref_batch_cond, "cond"):
        ref_batch = ref_batch_cond.cond
    else:
        ref_batch = ref_batch_cond

    # ref_batch should be list of encoded reference latents
    if not isinstance(ref_batch, list) or len(ref_batch) == 0:
        return sliced_vace

    # Get current window index from context_handler
    window_idx = getattr(context_handler, '_wva_current_window_idx', 0)
    num_refs = len(ref_batch)

    mapping_cond = new_cond_item.get("window_reference_mapping", None)
    mapping_str = None
    if mapping_cond is not None:
        mapping_str = mapping_cond.cond if hasattr(mapping_cond, "cond") else mapping_cond
    explicit_mapping = _parse_ref_mapping(mapping_str) if mapping_str else None

    # Get per-reference strengths
    strengths_cond = new_cond_item.get("window_reference_strengths", None)
    strengths_value = None
    if strengths_cond is not None:
        strengths_value = strengths_cond.cond if hasattr(strengths_cond, "cond") else strengths_cond

    if isinstance(strengths_value, (list, tuple)):
        explicit_strengths = list(strengths_value) if strengths_value else None
    elif isinstance(strengths_value, (int, float)):
        explicit_strengths = [float(strengths_value)]
    else:
        explicit_strengths = None

    if explicit_mapping:
        # Explicit mapping: use mapping[window_idx], repeat last if overflow
        if window_idx < len(explicit_mapping):
            selected_ref_idx = explicit_mapping[window_idx]
        else:
            selected_ref_idx = explicit_mapping[-1]
        selected_ref_idx = min(selected_ref_idx, num_refs - 1)
    else:
        # Default: 1-to-1 mapping, repeat last reference for extra windows
        if window_idx < num_refs:
            selected_ref_idx = window_idx
        else:
            selected_ref_idx = num_refs - 1

    selected_ref = ref_batch[selected_ref_idx]

    # Expected selected_ref shape: [1, 32, T_latent, H_latent, W_latent] where T_latent should be 1
    if selected_ref.ndim != 5:
        logging.warning(f"[WanVaceAdvanced] Reference tensor has unexpected ndim {selected_ref.ndim}, expected 5. Skipping injection.")
        return sliced_vace

    ref_channels = selected_ref.shape[1]
    if ref_channels != 32:
        logging.warning(f"[WanVaceAdvanced] Reference has {ref_channels} channels, expected {32}. Skipping injection.")
        return sliced_vace

    # sliced_vace shape: [B, num_ops, 96, T, H, W]
    vace_h, vace_w = sliced_vace.shape[-2], sliced_vace.shape[-1]
    ref_h, ref_w = selected_ref.shape[-2], selected_ref.shape[-1]

    if ref_h != vace_h or ref_w != vace_w:
        logging.warning(
            f"[WanVaceAdvanced] Reference spatial dims ({ref_h}x{ref_w}) don't match "
            f"vace_context ({vace_h}x{vace_w}). Skipping injection."
        )
        return sliced_vace

    sliced_vace = sliced_vace.clone()
    selected_ref_device = selected_ref.to(device)
    batch_size = sliced_vace.shape[0]
    num_ops = sliced_vace.shape[1]
    ref_temporal = selected_ref_device.shape[2]  # Should be 1

    # Determine strength for this reference
    if explicit_strengths:
        if len(explicit_strengths) == 1:
            ref_strength = explicit_strengths[0]
        elif selected_ref_idx < len(explicit_strengths):
            ref_strength = explicit_strengths[selected_ref_idx]
        else:
            ref_strength = 1.0
    else:
        ref_strength = 1.0

    # Expand reference to match vace_context dimensions
    # selected_ref: [1, 32, T_ref, H, W] -> [B, num_ops, 32, T_ref, H, W]
    selected_ref_expanded = selected_ref_device.unsqueeze(1).expand(
        batch_size, num_ops, -1, -1, -1, -1
    )

    if ref_strength != 1.0:
        selected_ref_expanded = selected_ref_expanded * ref_strength

    # Inject into the first frames of the inactive channels
    sliced_vace[:, :, :32, :ref_temporal, :, :] = selected_ref_expanded

    mapping_info = " (explicit mapping)" if explicit_mapping else ""
    logging.info(
        f"[WanVaceAdvanced] Window {window_idx}: injected reference [{selected_ref_idx}] "
        f"(strength={ref_strength:.2f}){mapping_info}"
    )

    return sliced_vace


def register_vace_context_windows_callbacks(model, context_handler):
    # Window tracking callback - called BEFORE each window
    # This stores the window_idx so RESIZE_COND_ITEM callbacks can access it
    tracker_callback = _create_window_tracker_callback(context_handler)
    comfy.patcher_extension.add_callback_with_key(
        comfy.context_windows.IndexListCallbacks.EVALUATE_CONTEXT_WINDOWS,
        "wanvaceadvanced_tracker",
        tracker_callback,
        context_handler.callbacks
    )

    # VACE resize callback - handles vace_context, vace_strength, and window references
    resize_callback = _create_vace_callback(context_handler)
    comfy.patcher_extension.add_callback_with_key(
        comfy.context_windows.IndexListCallbacks.RESIZE_COND_ITEM,
        "wanvaceadvanced",
        resize_callback,
        context_handler.callbacks
    )
