# SPDX-License-Identifier: MIT

import dataclasses
import inspect
import pkgutil
import re
from typing import Any, List, Optional

import clintrials


def extract_docstring_params(doc: Optional[str]) -> List[str]:
    """Extract parameter names from a Google-style or standard Python docstring.

    Args:
        doc (Optional[str]): The docstring of the function or class.

    Returns:
        List[str]: A list of parameter names found under the arguments block.
    """
    if not doc:
        return []
    lines = doc.split("\n")
    params = []
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "parameters:", "arguments:"):
            in_args = True
            continue
        if in_args:
            if stripped.lower() in ("returns:", "raises:", "examples:", "yields:", "warns:", "note:", "warning:", "type:"):
                in_args = False
                continue
            # Regex to match: parameter_name (type): description or parameter_name: description
            m = re.match(r"^\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]+\))?\s*:", line)
            if m:
                param_name = m.group(1)
                if param_name.lower() not in ("note", "example", "warning", "caution", "todo", "fixme"):
                    params.append(param_name)
    return params

def get_sig_params(obj: Any) -> Optional[List[str]]:
    """Retrieve parameter names from the signature of a given callable.

    Args:
        obj (Any): The function, method, or class object to inspect.

    Returns:
        Optional[List[str]]: A list of non-self, non-varargs, non-varkwargs parameters,
            or None if the signature cannot be retrieved.
    """
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return None
    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        params.append(name)
    return params

def test_docstring_parameter_signatures() -> None:
    """Validate that Google-style docstring parameter blocks match actual signatures.

    This test dynamically walks through all public non-test modules in the clintrials
    package, inspects every public function and class __init__ method, and asserts
    that their parameters match their documented arguments exactly.

    Raises:
        AssertionError: If any parameter mismatches or undocumented signatures are found.
    """
    mismatches = []
    for module_info in pkgutil.walk_packages(clintrials.__path__, clintrials.__name__ + "."):
        module_name = module_info.name
        if "test" in module_name:
            continue
        try:
            # Import module
            mod = __import__(module_name, fromlist=["*"])
        except Exception:
            continue
        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            obj_module = getattr(obj, "__module__", "")
            if obj_module != module_name:
                continue

            # Check function
            if inspect.isfunction(obj):
                sig_p = get_sig_params(obj)
                if sig_p:  # Has signature parameters
                    doc_p = extract_docstring_params(obj.__doc__)
                    if set(sig_p) != set(doc_p):
                        mismatches.append(f"Function {module_name}.{name} signature {sig_p} != docstring {doc_p}")
            # Check class __init__
            elif inspect.isclass(obj):
                if dataclasses.is_dataclass(obj):
                    continue
                init_func = getattr(obj, "__init__", None)
                if init_func:
                    init_mod = getattr(init_func, "__module__", "")
                    if init_mod.startswith("clintrials"):
                        sig_p = get_sig_params(init_func)
                        if sig_p:  # Has signature parameters
                            doc_p = extract_docstring_params(init_func.__doc__)
                            if set(sig_p) != set(doc_p):
                                mismatches.append(f"Class {module_name}.{obj.__name__}.__init__ signature {sig_p} != docstring {doc_p}")

    assert not mismatches, "Docstring-to-signature parameter name mismatches found:\n" + "\n".join(mismatches)
