# ruff: noqa: D100, D103, T201
import re

with open('tests/test_dashboard_views.py', 'r') as f:
    text = f.read()

# I will replace the git conflict markers with the combined version.
# For each conflict, I will use the HEAD version but also add monkeypatch.setattr(view, "st", st_mock) and importlib.reload(view) where appropriate.

new_text = re.sub(
    r'<<<<<<< HEAD\n\s*import clintrials\.utils as utils\n\s*import clintrials\.core\.simulation as sim\n\s*monkeypatch\.setattr\(utils, "ParameterSpace", MagicMock\(\)\)\n=======\n\s*monkeypatch\.setattr\(crm_view, "st", st_mock\)\n>>>>>>> origin/dedupe',
    r'import clintrials.utils as utils\n    import clintrials.core.simulation as sim\n    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())\n    monkeypatch.setattr(crm_view, "st", st_mock)',
    text
)

new_text = re.sub(
    r'<<<<<<< HEAD\n\s*st_mock = _make_streamlit_mock\(\)\n\s*monkeypatch\.setattr\(crm_view, "st", st_mock\)\n\s*monkeypatch\.setitem\(sys\.modules, "streamlit", st_mock\)\n\n\s*import clintrials\.utils as utils\n\s*import clintrials\.core\.simulation as sim\n\s*monkeypatch\.setattr\(utils, "ParameterSpace", MagicMock\(\)\)\n=======\n\s*st_mock = _make_streamlit_mock\(\)  # type: ignore\n\s*monkeypatch\.setitem\(sys\.modules, "streamlit", st_mock\)\n\s*importlib\.reload\(crm_view\)\n\n\s*monkeypatch\.setattr\(crm_view, "st", st_mock\)\n>>>>>>> origin/dedupe',
    r'st_mock = _make_streamlit_mock()\n    monkeypatch.setitem(sys.modules, "streamlit", st_mock)\n    import importlib\n    importlib.reload(crm_view)\n    monkeypatch.setattr(crm_view, "st", st_mock)\n    import clintrials.utils as utils\n    import clintrials.core.simulation as sim\n    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())',
    new_text
)

new_text = re.sub(
    r'<<<<<<< HEAD\n\s*import clintrials\.utils as utils\n\s*import clintrials\.core\.simulation as sim\n\s*monkeypatch\.setattr\(utils, "ParameterSpace", MagicMock\(\)\)\n=======\n\s*monkeypatch\.setattr\(efftox_view, "st", st_mock\)\n>>>>>>> origin/dedupe',
    r'import clintrials.utils as utils\n    import clintrials.core.simulation as sim\n    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())\n    monkeypatch.setattr(efftox_view, "st", st_mock)',
    new_text
)

new_text = re.sub(
    r'<<<<<<< HEAD\n\s*st_mock = _make_streamlit_mock\(\)\n\s*monkeypatch\.setattr\(efftox_view, "st", st_mock\)\n\s*monkeypatch\.setitem\(sys\.modules, "streamlit", st_mock\)\n\n\s*import clintrials\.utils as utils\n\s*import clintrials\.core\.simulation as sim\n\s*monkeypatch\.setattr\(utils, "ParameterSpace", MagicMock\(\)\)\n=======\n\s*st_mock = _make_streamlit_mock\(\)  # type: ignore\n\s*monkeypatch\.setitem\(sys\.modules, "streamlit", st_mock\)\n\s*importlib\.reload\(efftox_view\)\n\n\s*monkeypatch\.setattr\(efftox_view, "st", st_mock\)\n>>>>>>> origin/dedupe',
    r'st_mock = _make_streamlit_mock()\n    monkeypatch.setitem(sys.modules, "streamlit", st_mock)\n    import importlib\n    importlib.reload(efftox_view)\n    monkeypatch.setattr(efftox_view, "st", st_mock)\n    import clintrials.utils as utils\n    import clintrials.core.simulation as sim\n    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())',
    new_text
)

new_text = re.sub(
    r'<<<<<<< HEAD\n\s*import clintrials\.utils as utils\n\s*import clintrials\.core\.simulation as sim\n\s*monkeypatch\.setattr\(utils, "ParameterSpace", MagicMock\(\)\)\n=======\n\s*monkeypatch\.setattr\(watu_view, "st", st_mock\)\n>>>>>>> origin/dedupe',
    r'import clintrials.utils as utils\n    import clintrials.core.simulation as sim\n    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())\n    monkeypatch.setattr(watu_view, "st", st_mock)',
    new_text
)

new_text = re.sub(
    r'<<<<<<< HEAD\n\s*st_mock = _make_streamlit_mock\(\)\n\s*monkeypatch\.setattr\(watu_view, "st", st_mock\)\n\s*monkeypatch\.setitem\(sys\.modules, "streamlit", st_mock\)\n\n\s*import clintrials\.utils as utils\n\s*import clintrials\.core\.simulation as sim\n\s*monkeypatch\.setattr\(utils, "ParameterSpace", MagicMock\(\)\)\n=======\n\s*st_mock = _make_streamlit_mock\(\)  # type: ignore\n\s*monkeypatch\.setitem\(sys\.modules, "streamlit", st_mock\)\n\s*importlib\.reload\(watu_view\)\n\n\s*monkeypatch\.setattr\(watu_view, "st", st_mock\)\n>>>>>>> origin/dedupe',
    r'st_mock = _make_streamlit_mock()\n    monkeypatch.setitem(sys.modules, "streamlit", st_mock)\n    import importlib\n    importlib.reload(watu_view)\n    monkeypatch.setattr(watu_view, "st", st_mock)\n    import clintrials.utils as utils\n    import clintrials.core.simulation as sim\n    monkeypatch.setattr(utils, "ParameterSpace", MagicMock())',
    new_text
)

with open('tests/test_dashboard_views.py', 'w') as f:
    f.write(new_text)

