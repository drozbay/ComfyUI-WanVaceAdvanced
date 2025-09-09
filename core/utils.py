DEBUG_ENABLED = True

def wan_print(*args, debug=False, **kwargs):
    if debug and not DEBUG_ENABLED:
        return
    
    if args:
        first_arg = str(args[0])
        if not first_arg.startswith('[WanVaceAdvanced]'):
            args = (f'[WanVaceAdvanced] {first_arg}',) + args[1:]
    
    print(*args, **kwargs)