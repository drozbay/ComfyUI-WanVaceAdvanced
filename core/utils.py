DEBUG_ENABLED = True

def wan_print(*args, debug=False, **kwargs):
    if debug and not DEBUG_ENABLED:
        return
    
    if args:
        first_arg = str(args[0])
        if not first_arg.startswith('[WanVaceAdvanced]'):
            args = (f'[WanVaceAdvanced] {first_arg}',) + args[1:]
    
    print(*args, **kwargs)

class WVAOptions:
    def __init__(self, use_tiled_vae=False):
        self.use_tiled_vae = use_tiled_vae
    
    def get_option(self, key, default=None):
        return getattr(self, key, default)
    
    def set_option(self, key, value):
        setattr(self, key, value)
        return self
    
    def __str__(self):
        return f"WVAOptions(use_tiled_vae={self.use_tiled_vae})"