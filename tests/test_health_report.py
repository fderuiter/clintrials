import os

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

def test_dashboard_framework_dependencies():  # type: ignore
    """Verify that Streamlit is the only dashboard framework, and Dash is removed."""
    pyproject_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    deps = config.get("tool", {}).get("poetry", {}).get("dependencies", {})
    assert "streamlit" in deps, "Streamlit must be listed as a dependency."
    assert "dash" not in deps, "Dash must NOT be listed as a dependency."

def test_dashboard_entry_point():  # type: ignore
    """Verify that the documented Streamlit entry point exists."""
    entry_point_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "clintrials", "visualization", "dashboard", "main.py"
    )
    assert os.path.exists(entry_point_path), f"Dashboard entry point missing: {entry_point_path}"
