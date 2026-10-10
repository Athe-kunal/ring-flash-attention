import importlib.util
import sys
import types
from pathlib import Path


def load_ring_flash_attn_module(name):
    """Load a ring_flash_attn module without importing flash_attn via __init__."""
    package_dir = Path(__file__).resolve().parents[1] / "ring_flash_attn"
    package_name = "ring_flash_attn"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    module_name = f"{package_name}.{name}"
    spec = importlib.util.spec_from_file_location(
        module_name, package_dir / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    triton_lens = load_ring_flash_attn_module("triton_lens")
    triton_lens.launch_visualizer(share=False)


if __name__ == "__main__":
    main()
