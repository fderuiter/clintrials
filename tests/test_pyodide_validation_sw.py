# SPDX-License-Identifier: MIT

from pathlib import Path


def test_service_worker_caching_configuration() -> None:
    """Verify that hub/sw.js correctly caches external CDNs, contains fallback, and lists offline files."""
    root = Path(__file__).parent.parent
    sw_file = root / "hub" / "sw.js"
    assert sw_file.exists(), "hub/sw.js should exist"

    sw_js_text = sw_file.read_text()

    # Verify that the offline files are listed in urlsToCache
    assert "clintrials-0.1.4-py3-none-any.whl" in sw_js_text, (
        "Service Worker must list clintrials-0.1.4-py3-none-any.whl in urlsToCache"
    )
    assert "runner.py" in sw_js_text, (
        "Service Worker must list runner.py in urlsToCache"
    )
    assert "schema.json" in sw_js_text, (
        "Service Worker must list schema.json in urlsToCache"
    )

    # Verify allowed CDN domains are present
    jsdelivr_domain = ".".join(["cdn", "jsdelivr", "net"])
    plotly_domain = ".".join(["cdn", "plot", "ly"])
    assert jsdelivr_domain in sw_js_text, (
        "Service Worker must support jsdelivr CDN caching"
    )
    assert plotly_domain in sw_js_text, (
        "Service Worker must support plotly CDN caching"
    )

    # Verify the fallback mechanism for missing offline wheels
    assert "Fallback / revert to known stable local package" in sw_js_text, (
        "Service Worker must have a fallback description comment"
    )
    assert "url.pathname.endsWith('.whl')" in sw_js_text, (
        "Service Worker must detect wheel requests for fallback"
    )


def test_background_worker_validation() -> None:
    """Verify that hub/worker.js implements PEP 440 package version verification."""
    root = Path(__file__).parent.parent
    worker_file = root / "hub" / "worker.js"
    assert worker_file.exists(), "hub/worker.js should exist"

    content = worker_file.read_text()

    # Verify message handler for validate_version
    assert 'type === "validate_version"' in content, (
        "worker.js must listen for validate_version events"
    )
    assert 'validate_version(' in content, (
        "worker.js must call the native validate_version Python function"
    )
    assert 'validate_version_result' in content, (
        "worker.js must post message back with validate_version_result"
    )


def test_index_html_settings_and_validation() -> None:
    """Verify that hub/index.html includes the Workspace Settings pane and triggers validation."""
    root = Path(__file__).parent.parent
    index_file = root / "hub" / "index.html"
    assert index_file.exists(), "hub/index.html should exist"

    content = index_file.read_text()

    # Verify Workspace Settings Section is present
    assert "Workspace Settings" in content, (
        "index.html must display the Workspace Settings section"
    )
    assert 'id="dependency-version"' in content, (
        "index.html must have the dependency-version input field"
    )

    # Verify client application handles dynamic validation trigger and errors
    assert 'validate_version' in content, (
        "index.html must trigger version validation on input/init"
    )
    assert 'validate_version_result' in content, (
        "index.html must handle the worker's validate_version_result"
    )
    assert 'dependency-version-error' in content, (
        "index.html must display errors under dependency-version-error"
    )
