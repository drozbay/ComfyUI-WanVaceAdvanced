import importlib.util
import sys
from pathlib import Path
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def load_module(module_name, file_path, package_name=None):
    """Loads a module from a .py file with proper package context."""
    spec = importlib.util.spec_from_file_location(module_name, file_path, submodule_search_locations=[])
    if package_name:
        spec.submodule_search_locations = [str(Path(file_path).parent.parent)]
    module = importlib.util.module_from_spec(spec)
    
    # Set up the module in sys.modules before execution to handle relative imports
    sys.modules[module_name] = module
    if package_name:
        module.__package__ = package_name
    
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        # Clean up on failure
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise

def load_nodes():
    """Loads nodes from the nodes folder."""
    # Get the package name for this custom node
    package_name = __name__
    
    # Ensure the package structure is in sys.modules
    if package_name not in sys.modules:
        sys.modules[package_name] = sys.modules[__name__]
    
    nodes_path = Path(__file__).parent / 'nodes'
    for file in nodes_path.glob("*.py"):
        if file.name != "__init__.py":
            try:
                module_name = f"{package_name}.nodes.{file.stem}"
                module = load_module(module_name, str(file), f"{package_name}.nodes")
                NODE_CLASS_MAPPINGS.update(getattr(module, 'NODE_CLASS_MAPPINGS', {}))
                NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', {}))
            except Exception as e:
                print(f"Error loading {file.name}: {e}\n{traceback.format_exc()}")

# Execute functions
load_nodes()

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]