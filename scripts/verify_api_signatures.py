import inspect
import json
import importlib
import sys
import argparse
from pathlib import Path

def get_signature_info(obj):
    try:
        sig = inspect.signature(obj)
        params = []
        for name, param in sig.parameters.items():
            if name.startswith('_'):
                continue
            params.append(name)
        return params
    except (ValueError, TypeError):
        return []

def get_public_methods(cls):
    methods = {}
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith('_') or name == '__init__':
            methods[name] = get_signature_info(method)
    return methods

def scan_module(module_name):
    mod = importlib.import_module(module_name)
    exports = getattr(mod, '__all__', [])
    if not exports:
        exports = [n for n in dir(mod) if not n.startswith('_')]
    
    manifest = {}
    for name in exports:
        if name.startswith('_'): continue
        obj = getattr(mod, name, None)
        if obj is None: continue
        
        if inspect.isclass(obj):
            manifest[name] = {
                "type": "class",
                "methods": get_public_methods(obj)
            }
        elif inspect.isfunction(obj):
            manifest[name] = {
                "type": "function",
                "parameters": get_signature_info(obj)
            }
    return manifest

def generate_manifest():
    modules = [
        'clintrials.core',
        'clintrials.dosefinding',
        'clintrials.winratio',
        'clintrials.visualization'
    ]
    manifest = {}
    for m in modules:
        manifest[m] = scan_module(m)
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Automated Package-Wide JSON Manifest Hook")
    parser.add_argument('--generate', action='store_true', help='Recreate or update the baseline JSON manifest file')
    parser.add_argument('--manifest', default='api_manifest.json', help='Path to the manifest file')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    current_manifest = generate_manifest()

    if args.generate:
        with open(manifest_path, 'w') as f:
            json.dump(current_manifest, f, indent=2)
        print(f"Manifest successfully generated and saved to {manifest_path}")
        sys.exit(0)
    
    if not manifest_path.exists():
        print(f"Error: Manifest file {manifest_path} does not exist. Run with --generate to create it.")
        sys.exit(1)
        
    with open(manifest_path, 'r') as f:
        baseline_manifest = json.load(f)
        
    diffs = []
    
    # Compare baseline and current
    for module_name, baseline_exports in baseline_manifest.items():
        if module_name not in current_manifest:
            diffs.append(f"Module '{module_name}' is missing.")
            continue
            
        current_exports = current_manifest[module_name]
        
        for name, baseline_obj in baseline_exports.items():
            if name not in current_exports:
                diffs.append(f"Export '{name}' missing in module '{module_name}'.")
                continue
                
            current_obj = current_exports[name]
            
            if baseline_obj['type'] != current_obj['type']:
                diffs.append(f"Export '{name}' type changed from {baseline_obj['type']} to {current_obj['type']}.")
                continue
                
            if baseline_obj['type'] == 'class':
                for method, params in baseline_obj['methods'].items():
                    if method not in current_obj['methods']:
                        diffs.append(f"Method '{method}' missing in class '{name}'.")
                        continue
                    if params != current_obj['methods'][method]:
                        diffs.append(f"Parameters for method '{name}.{method}' changed from {params} to {current_obj['methods'][method]}.")
            
            elif baseline_obj['type'] == 'function':
                if baseline_obj['parameters'] != current_obj['parameters']:
                    diffs.append(f"Parameters for function '{name}' changed from {baseline_obj['parameters']} to {current_obj['parameters']}.")
                    
    # Also check for newly added things not in baseline
    for module_name, current_exports in current_manifest.items():
        baseline_exports = baseline_manifest.get(module_name, {})
        for name, current_obj in current_exports.items():
            if name not in baseline_exports:
                diffs.append(f"Export '{name}' is newly added to module '{module_name}'.")
                continue
                
            baseline_obj = baseline_exports[name]
            if baseline_obj['type'] == 'class':
                for method, params in current_obj['methods'].items():
                    if method not in baseline_obj['methods']:
                        diffs.append(f"Method '{method}' is newly added to class '{name}'.")
                        
    if diffs:
        print("API Signature Mismatch Detected!")
        print("The following differences were found compared to the baseline:")
        for diff in diffs:
            print(f" - {diff}")
        print("\nIf these changes are intentional, run 'poetry run python scripts/verify_api_signatures.py --generate' to update the baseline.")
        sys.exit(1)
        
    print("API Signatures match the baseline.")
    sys.exit(0)

if __name__ == '__main__':
    main()
