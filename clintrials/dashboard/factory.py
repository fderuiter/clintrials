import inspect

import streamlit as st


def _build_registry():
    registry = {}

    # Extract from Win Ratio core logic
    try:
        from clintrials.winratio.main import run_simulation

        doc = run_simulation.__doc__
        if doc:
            lines = doc.split("\n")
            for i, line in enumerate(lines):
                line = line.strip()
                if "(" in line and "):" in line:
                    var_name = line.split(" (")[0].strip()
                    desc = line.split("):", 1)[1].strip()

                    # Check next line for continuation
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if (
                            next_line
                            and not ("(" in next_line and "):" in next_line)
                            and not next_line.startswith(("Returns:", "Args:"))
                        ):
                            desc += " " + next_line

                    registry[var_name] = desc
    except ImportError:
        pass

    # Main dashboard variables (fallback if not in docstrings)
    registry["design_type"] = "Select the type of trial design."
    registry["uploaded_file"] = "Upload a JSON file with simulation results."
    registry["run_simulation_button"] = (
        "Run a Monte Carlo simulation to estimate win-ratio power."
    )

    return registry


REGISTRY = _build_registry()


def create_widget(st_module, widget_type, var_name, *args, **kwargs):
    """
    Factory function to create a Streamlit widget with an automatically
    applied help text based on the variable name.
    """
    if var_name in REGISTRY:
        kwargs["help"] = REGISTRY[var_name]

    if widget_type == "selectbox":
        return st_module.sidebar.selectbox(*args, **kwargs)
    elif widget_type == "file_uploader":
        return st_module.sidebar.file_uploader(*args, **kwargs)
    elif widget_type == "number_input":
        return st_module.sidebar.number_input(*args, **kwargs)
    elif widget_type == "button":
        return st_module.sidebar.button(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported widget type: {widget_type}")
