import torch
import comfy.patcher_extension
import comfy.context_windows


def vace_resize_cond_item_callback(cond_key, cond_value, window, x_in, device, new_cond_item):
    dim = 2
    # Handle vace_context
    if cond_key == "vace_context":
        if not (hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor)):
            return None

        vace_cond = cond_value.cond
        if vace_cond.ndim <= 3:
            return None

        x_temporal = x_in.size(dim)
        vace_temporal = vace_cond.size(3)

        # Detect phantom (time_dim_concat present AND vace_context extended)
        has_time_dim_concat = "time_dim_concat" in new_cond_item
        has_extended_vace = vace_temporal > x_temporal

        if has_time_dim_concat and has_extended_vace:
            # Phantom mode: split, slice main, re-append phantom
            T_phantom = vace_temporal - x_temporal
            vace_main = vace_cond[:, :, :, :-T_phantom, :, :]
            vace_phantom = vace_cond[:, :, :, -T_phantom:, :, :].to(device)
            sliced_main = window.get_tensor(vace_main, device, dim=3)

            # Inject reference for non-zero-start windows
            if window.index_list[0] != 0:
                sliced_main = sliced_main.clone()
                reference_latent = vace_main[:, :, :32, :1, :, :].to(device)
                sliced_main[:, :, :32, :1, :, :] = reference_latent

            sliced_vace = torch.cat([sliced_main, vace_phantom], dim=3)
            return cond_value._copy_with(sliced_vace)

        elif vace_temporal == x_temporal:
            # Standard VACE (no phantom)
            sliced_vace = window.get_tensor(vace_cond, device, dim=3)

            # Inject reference for non-zero-start windows
            if window.index_list[0] != 0:
                sliced_vace = sliced_vace.clone()
                reference_latent = vace_cond[:, :, :32, :1, :, :].to(device)
                sliced_vace[:, :, :32, :1, :, :] = reference_latent

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

        # Flat list of numbers
        if isinstance(inner_list[0], (int, float)):
            if len(inner_list) == x_temporal:
                return cond_value._copy_with([inner_list[i] for i in window.index_list])
            return None

        # List of lists (per-frame strengths)
        if isinstance(inner_list[0], list):
            sliced_list = []
            for sub_list in inner_list:
                if len(sub_list) > x_temporal:
                    # Phantom: slice main portion, preserve phantom at end
                    T_phantom = len(sub_list) - x_temporal
                    main_strengths = sub_list[:-T_phantom]
                    phantom_strengths = sub_list[-T_phantom:]
                    sliced_main = [main_strengths[i] for i in window.index_list]
                    sliced_list.append(sliced_main + phantom_strengths)
                elif len(sub_list) == x_temporal:
                    sliced_list.append([sub_list[i] for i in window.index_list])
                else:
                    sliced_list.append(sub_list)
            return cond_value._copy_with(sliced_list)

        return None

    # Handle vace_frames, vace_mask (list of tensors)
    if cond_key in ("vace_frames", "vace_mask"):
        if not isinstance(cond_value, list) or len(cond_value) == 0:
            return None
        if not isinstance(cond_value[0], torch.Tensor):
            return None

        sliced_list = []
        for tensor in cond_value:
            if dim < tensor.ndim and tensor.size(dim) == x_in.size(dim):
                sliced_list.append(window.get_tensor(tensor, device, dim=dim))
            else:
                sliced_list.append(tensor.to(device) if device else tensor)
        return sliced_list

    return None  # Not handled


def register_vace_context_windows_callbacks(model, context_handler):
    comfy.patcher_extension.add_callback_with_key(
        comfy.context_windows.IndexListCallbacks.RESIZE_COND_ITEM,
        "wanvaceadvanced",
        vace_resize_cond_item_callback,
        context_handler.callbacks
    )
