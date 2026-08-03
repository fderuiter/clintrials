#!/usr/bin/env python3
"""Script to extract Python public API documentation and resolve dynamic docstrings into a JSON manifest."""

import inspect
import json
import os
import pkgutil
import re
from typing import Any, Dict, List, Optional

import clintrials


def parse_docstring_to_params(doc: Optional[str]) -> List[Dict[str, Any]]:
    """Parse a docstring to extract parameters following the specified rules."""
    if not doc:
        return []
    lines = doc.split("\n")
    params = []
    in_args = False
    current_param = None

    terminators = {"returns:", "raises:", "examples:", "yields:", "warns:", "note:", "warning:", "type:"}
    exclude_keywords = {"note", "example", "warning", "caution", "todo", "fixme"}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.lower() in ("args:", "parameters:", "arguments:"):
            in_args = True
            current_param = None
            continue

        if in_args:
            if stripped.lower() in terminators:
                in_args = False
                current_param = None
                continue

            # Match: parameter_name (type): description or parameter_name: description
            # Capture Group 1: Name, Group 2: Type (optional), Group 3: Description (optional)
            m = re.match(r"^\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(([^)]+)\))?\s*:\s*(.*)", line)
            if m:
                param_name = m.group(1)
                param_type = m.group(2) or ""
                param_desc = m.group(3).strip()

                if param_name.lower() in exclude_keywords:
                    continue

                current_param = {
                    "name": param_name,
                    "type": param_type,
                    "description": param_desc
                }
                params.append(current_param)
            else:
                # Part of previous parameter description if indented
                if current_param and (line.startswith(" ") or line.startswith("\t")):
                    if current_param["description"]:
                        current_param["description"] += " " + stripped
                    else:
                        current_param["description"] = stripped

    return params


def get_signature_info(obj: Any) -> Dict[str, Any]:
    """Retrieve detailed parameter signature information including defaults."""
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return {}

    params_info = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        default_val = None
        if param.default is not inspect.Parameter.empty:
            default_val = repr(param.default)

        annotation_str = ""
        if param.annotation is not inspect.Parameter.empty:
            if hasattr(param.annotation, "__name__"):
                annotation_str = param.annotation.__name__
            else:
                annotation_str = str(param.annotation)

        params_info[name] = {
            "name": name,
            "default": default_val,
            "kind": str(param.kind),
            "annotation": annotation_str
        }
    return params_info


def extract_parameters_metadata(obj: Any) -> List[Dict[str, Any]]:
    """Combine docstring-parsed parameters with actual signature default values."""
    parsed_params = parse_docstring_to_params(obj.__doc__)
    sig_info = get_signature_info(obj)

    for p in parsed_params:
        p_name = p["name"]
        if p_name in sig_info:
            p["default"] = sig_info[p_name]["default"]
            if not p["type"] and sig_info[p_name]["annotation"]:
                p["type"] = sig_info[p_name]["annotation"]
        else:
            p["default"] = None

    return parsed_params


def extract_class_metadata(cls_obj: Any) -> Dict[str, Any]:
    """Extract class metadata including its docstring, __init__ parameters, and public methods."""
    cls_name = cls_obj.__name__
    cls_doc = cls_obj.__doc__ or ""

    init_func = getattr(cls_obj, "__init__", None)
    parameters = []
    if init_func:
        parameters = extract_parameters_metadata(init_func)

    methods = []
    for name in sorted(dir(cls_obj)):
        if name.startswith("_"):
            continue
        method_obj = getattr(cls_obj, name)
        if inspect.isroutine(method_obj):
            method_module = getattr(method_obj, "__module__", "") or ""
            if not method_module.startswith("clintrials"):
                continue

            sig_str = ""
            try:
                sig_str = str(inspect.signature(method_obj))
            except (ValueError, TypeError):
                pass

            methods.append({
                "name": name,
                "signature": sig_str,
                "docstring": method_obj.__doc__ or "",
                "parameters": extract_parameters_metadata(method_obj)
            })

    return {
        "name": cls_name,
        "docstring": cls_doc,
        "parameters": parameters,
        "methods": methods
    }


def main() -> None:
    """Discovers all public submodules, extracts classes, functions, and parameters, and outputs a manifest JSON."""
    manifest = {"modules": {}}

    # Discover and parse all public modules
    modules_to_process = []

    # Process package root if needed, but it only exports submodules
    for module_info in pkgutil.walk_packages(clintrials.__path__, clintrials.__name__ + "."):
        module_name = module_info.name
        if "test" in module_name:
            continue
        modules_to_process.append(module_name)

    # Sort modules for deterministic output
    modules_to_process.sort()

    for module_name in modules_to_process:
        try:
            mod = __import__(module_name, fromlist=["*"])
        except Exception as e:
            print(f"Skipping module {module_name} due to import error: {e}")  # noqa: T201
            continue

        mod_doc = mod.__doc__ or ""
        classes = []
        functions = []

        # Find items defined inside this specific module
        for name in sorted(dir(mod)):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)

            # Check module ownership
            obj_module = getattr(obj, "__module__", "")
            if obj_module != module_name:
                continue

            if inspect.isclass(obj):
                classes.append(extract_class_metadata(obj))
            elif inspect.isfunction(obj):
                sig_str = ""
                try:
                    sig_str = str(inspect.signature(obj))
                except (ValueError, TypeError):
                    pass

                functions.append({
                    "name": name,
                    "signature": sig_str,
                    "docstring": obj.__doc__ or "",
                    "parameters": extract_parameters_metadata(obj)
                })

        manifest["modules"][module_name] = {
            "name": module_name,
            "docstring": mod_doc,
            "classes": classes,
            "functions": functions
        }

    # Write intermediate manifest JSON
    manifest_path = os.path.abspath("docs_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated API documentation manifest at {manifest_path}")  # noqa: T201


if __name__ == "__main__":
    main()
