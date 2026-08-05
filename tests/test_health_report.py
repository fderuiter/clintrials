import os

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef, unused-ignore]

def test_dashboard_framework_dependencies():
    """Verify that Streamlit is the only dashboard framework, and Dash is removed."""
    pyproject_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    project = config.get("project", {})
    deps = project.get("dependencies", [])
    optional_deps = project.get("optional-dependencies", {})

    all_deps = list(deps)
    for ext_deps in optional_deps.values():
        all_deps.extend(ext_deps)

    has_streamlit = any(d.strip().startswith("streamlit") for d in all_deps)
    has_dash = any(d.strip().startswith("dash") for d in all_deps)

    assert has_streamlit, "Streamlit must be listed as a dependency."
    assert not has_dash, "Dash must NOT be listed as a dependency."

def test_dashboard_entry_point():
    """Verify that the documented Streamlit entry point exists."""
    entry_point_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "clintrials", "visualization", "dashboard", "main.py"
    )
    assert os.path.exists(entry_point_path), f"Dashboard entry point missing: {entry_point_path}"
