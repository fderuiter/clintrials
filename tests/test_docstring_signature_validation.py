# SPDX-License-Identifier: MIT

import inspect
import pkgutil
import re
from typing import Any, List, Optional

import clintrials


def extract_docstring_params(doc: Optional[str]) -> List[str]:
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
            if stripped.lower() in (
                "returns:",
                "raises:",
                "examples:",
                "yields:",
                "warns:",
                "note:",
                "warning:",
                "type:",
            ):
                in_args = False
                continue
            # Regex to match: parameter_name (type): description or parameter_name: description
            m = re.match(r"^\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]+\))?\s*:", line)
            if m:
                param_name = m.group(1)
                if param_name.lower() not in (
                    "note",
                    "example",
                    "warning",
                    "caution",
                    "todo",
                    "fixme",
                ):
                    params.append(param_name)
    return params


def get_sig_params(obj: Any) -> Optional[List[str]]:
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return None
    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        params.append(name)
    return params


def test_docstring_parameter_signatures() -> None:
    mismatches = []
    for module_info in pkgutil.walk_packages(
        clintrials.__path__, clintrials.__name__ + "."
    ):
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
            if not obj_module.startswith("clintrials"):
                continue

            # Check function
            if inspect.isfunction(obj):
                sig_p = get_sig_params(obj)
                doc_p = extract_docstring_params(obj.__doc__)
                if sig_p is not None and doc_p:
                    # check if they match
                    if set(sig_p) != set(doc_p):
                        mismatches.append(
                            f"Function {module_name}.{name} signature {sig_p} != docstring {doc_p}"
                        )
            # Check class __init__
            elif inspect.isclass(obj):
                init_func = getattr(obj, "__init__", None)
                if init_func:
                    init_mod = getattr(init_func, "__module__", "")
                    if init_mod.startswith("clintrials"):
                        sig_p = get_sig_params(init_func)
                        doc_p = extract_docstring_params(init_func.__doc__)
                        if sig_p is not None and doc_p:
                            if set(sig_p) != set(doc_p):
                                mismatches.append(
                                    f"Class {module_name}.{obj.__name__}.__init__ signature {sig_p} != docstring {doc_p}"
                                )

    assert not mismatches, (
        "Docstring-to-signature parameter name mismatches found:\n"
        + "\n".join(mismatches)
    )
