import importlib
import os
import sys

import pytest

# Ensure the project root is on sys.path so local 'clintrials' package is used
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clintrials.core.registry import PROTOCOL_REGISTRY

_original_reload = importlib.reload

@pytest.fixture(autouse=True)
def isolate_dashboard_tests():
    """
    Automatically captures and restores the global registry state,
    and identifies/reloads modules that were reloaded with mock UI libraries.
    """
    reloaded_modules = set()

    def _mock_reload(module):
        reloaded_modules.add(module)
        return _original_reload(module)

    importlib.reload = _mock_reload

    # Snapshot registry
    if hasattr(PROTOCOL_REGISTRY, "snapshot"):
        PROTOCOL_REGISTRY.snapshot()

    yield

    importlib.reload = _original_reload

    # Restore registry
    if hasattr(PROTOCOL_REGISTRY, "restore"):
        PROTOCOL_REGISTRY.restore()

    # Reload modules that were reloaded during the test.
    # Because this autouse fixture does not depend on `monkeypatch`,
    # its teardown runs AFTER monkeypatch has fully restored module variables
    # to their (incorrect) mocked state. Reloading them now correctly restores
    # them to their actual dependencies.
    for mod in reloaded_modules:
        _original_reload(mod)
