


def _build_registry():
    registry = {}

    def extract_from_docstring(doc, aliases=None):
        if not doc:
            return
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
                        and not next_line.startswith(
                            ("Returns:", "Args:", "Raises:", "Yields:")
                        )
                    ):
                        desc += " " + next_line

                registry[var_name] = desc
                if aliases and var_name in aliases:
                    for alias in aliases[var_name]:
                        registry[alias] = desc

    # Extract from Win Ratio core logic
    try:
        from clintrials.winratio.main import run_simulation

        extract_from_docstring(run_simulation.__doc__)
    except ImportError:
        pass

    # Extract from CRM core logic
    try:
        from clintrials.dosefinding import simulate_dose_finding_trial
        from clintrials.dosefinding.crm import CRM

        extract_from_docstring(CRM.__init__.__doc__)
        extract_from_docstring(
            simulate_dose_finding_trial.__doc__,
            aliases={"true_toxicities": ["true_tox"]},
        )
    except ImportError:
        pass

    # Extract from EffTox core logic
    try:
        from clintrials.dosefinding.efficacytoxicity import simulate_trial
        from clintrials.dosefinding.efftox import EffTox

        extract_from_docstring(EffTox.__init__.__doc__)
        extract_from_docstring(
            simulate_trial.__doc__,
            aliases={
                "true_toxicities": ["true_prob_tox"],
                "true_efficacies": ["true_prob_eff"],
            },
        )
    except ImportError:
        pass

    # Extract from WATU core logic
    try:
        from clintrials.dosefinding.watu import WATU

        extract_from_docstring(WATU.__init__.__doc__)
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


def render_metric(st_module, label, value, precision=4):
    """
    Renders a semantic metric card with configurable numeric precision for statistical floats.
    """
    if isinstance(value, float):
        formatted_value = f"{value:.{precision}f}"
    elif isinstance(value, (list, tuple)) and all(
        isinstance(x, (int, float)) for x in value
    ):
        formatted_value = "(" + ", ".join(f"{x:.{precision}f}" for x in value) + ")"
    else:
        formatted_value = str(value)

    st_module.metric(label=label, value=formatted_value)


def render_accessible_chart(st_module, fig, expander_label="Data Summary"):
    """
    Shared utility to render a Plotly chart with an accessible Markdown table summary.
    """
    meta = getattr(getattr(fig, "layout", None), "meta", "No data summary available.")
    
    if hasattr(fig, "layout") and hasattr(fig.layout, "meta"):
        fig.layout.meta = None
    st_module.plotly_chart(fig)

    if hasattr(meta, "html"):
        st_module.markdown(meta.html, unsafe_allow_html=True)
    else:
        st_module.markdown(meta)
