import importlib
from typing import Any

def instantiate_from_string(path: str, *args, **kwargs) -> Any:
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(*args, **kwargs)