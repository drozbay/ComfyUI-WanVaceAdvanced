"""HuMo I2V Patch Node - enables I2V support for WAN21_HuMo 17B model."""

from ..core.patches import patch_humo_i2v_support, is_wan21_humo_model
from ..core.utils import wan_print

class HuMoI2VPatch:
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

    def patch_model(self, model):
        if not is_wan21_humo_model(model):
            wan_print(f"Warning: Model type '{type(model.model).__name__}' is not WAN21_HuMo")
            return (model,)

        m = model.clone()

        try:
            m = patch_humo_i2v_support(m)
        except Exception as e:
            wan_print(f"Error applying HuMo I2V patch: {e}")
            return (model,)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "HuMoI2VPatch": HuMoI2VPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuMoI2VPatch": "HuMo I2V Patch",
}
