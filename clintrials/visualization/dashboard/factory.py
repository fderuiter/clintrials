


class ScopedUIRegistry(dict):
    def __init__(self):
        super().__init__()
        self._namespaces = {}

    def set_help(self, namespace, var_name, desc):
        if namespace not in self._namespaces:
            self._namespaces[namespace] = {}
        self._namespaces[namespace][var_name] = desc
        # Fallback flat dict for backward compatibility
        self[var_name] = desc

    def get_help(self, var_name, design_type=None):
        if design_type and design_type in self._namespaces:
            if var_name in self._namespaces[design_type]:
                return self._namespaces[design_type][var_name]

        # Fallback to global namespace
        if "global" in self._namespaces and var_name in self._namespaces["global"]:
            return self._namespaces["global"][var_name]

        # Fallback to flat dictionary
        return self.get(var_name)


def _build_registry():
    registry = ScopedUIRegistry()

    def extract_from_docstring(doc, namespace="global", aliases=None):
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

                registry.set_help(namespace, var_name, desc)
                if aliases and var_name in aliases:
                    for alias in aliases[var_name]:
                        registry.set_help(namespace, alias, desc)

    # Extract from Win Ratio core logic
    try:
        from clintrials.winratio.main import run_simulation

        extract_from_docstring(run_simulation.__doc__, namespace="Win Ratio")
    except ImportError:
        pass

    # Extract from CRM core logic
    try:
        from clintrials.dosefinding import simulate_dose_finding_trial
        from clintrials.dosefinding.crm import CRM

        extract_from_docstring(CRM.__init__.__doc__, namespace="CRM")
        extract_from_docstring(
            simulate_dose_finding_trial.__doc__,
            namespace="CRM",
            aliases={"true_toxicities": ["true_tox"]},
        )
    except ImportError:
        pass

    # Extract from EffTox core logic
    try:
        from clintrials.dosefinding.efficacytoxicity import simulate_trial
        from clintrials.dosefinding.efftox import EffTox

        extract_from_docstring(EffTox.__init__.__doc__, namespace="EffTox")
        extract_from_docstring(
            simulate_trial.__doc__,
            namespace="EffTox",
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

        extract_from_docstring(WATU.__init__.__doc__, namespace="WATU")
    except ImportError:
        pass

    # Main dashboard variables (fallback if not in docstrings)
    registry.set_help("global", "design_type", "Select the type of trial design.")
    registry.set_help("global", "uploaded_file", "Upload a JSON file with simulation results.")
    registry.set_help("global", "run_simulation_button", "Run a Monte Carlo simulation to estimate win-ratio power.")

    return registry


UI_REGISTRY = _build_registry()


def create_widget(st_module, widget_type, var_name, *args, **kwargs):
    """Factory function to create a Streamlit widget with an automatically
    applied help text based on the variable name.
    """
    design_type = kwargs.pop("design_type", None)
    if not design_type and hasattr(st_module, "session_state"):
        design_type = st_module.session_state.get("design_type")

    help_text = UI_REGISTRY.get_help(var_name, design_type)
    if help_text:
        kwargs["help"] = help_text

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
    """Renders a semantic metric card with configurable numeric precision for statistical floats.
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
    """Shared utility to render a Plotly chart with an accessible Markdown table summary.
    """
    meta = getattr(getattr(fig, "layout", None), "meta", "No data summary available.")

    if hasattr(fig, "layout") and hasattr(fig.layout, "meta"):
        fig.layout.meta = None
    st_module.plotly_chart(fig)

    if hasattr(meta, "html"):
        st_module.markdown(meta.html, unsafe_allow_html=True)
    else:
        st_module.markdown(meta)
