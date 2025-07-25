import importlib.util
import importlib.machinery
import sys
import os

def find_module(name, path=None):
    if path is None:
        path = sys.path
    spec = importlib.machinery.PathFinder.find_spec(name, path)
    if spec is None:
        raise ImportError(f"Cannot find module {name}")
    filename = spec.origin
    if filename is None:
        raise ImportError(f"Cannot find module {name}")
    file = open(filename, 'rb')
    suffix = os.path.splitext(filename)[1]
    mode = file.mode
    return file, filename, (suffix, mode, 0)

def load_module(name, file, filename, desc):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def acquire_lock():
    pass

def release_lock():
    pass
