from pathlib import Path


def test_service_worker_scoping_and_kill_switch() -> None:
    root = Path(__file__).parent.parent

    # 1. Verify docs/_static/custom.js unregisters old root-scoped SWs
    custom_js = root / "docs" / "_static" / "custom.js"
    assert custom_js.exists()
    content_custom = custom_js.read_text()

    assert "navigator.serviceWorker.register" not in content_custom, "custom.js should not register service workers"
    assert "getRegistrations" in content_custom, "custom.js should inspect active registrations"
    assert "unregister" in content_custom, "custom.js should unregister old root service workers"
    assert "caches.delete('sim-hub-cache-v4')" in content_custom, "custom.js should delete the stale cache"

    # 2. Verify docs/_extra/sw.js is a self-unregistering kill-switch
    sw_extra = root / "docs" / "_extra" / "sw.js"
    assert sw_extra.exists()
    content_extra = sw_extra.read_text()
    assert "caches.delete" in content_extra, "kill-switch should clear cache"
    assert "registration.unregister" in content_extra, "kill-switch should unregister itself"

    # 3. Verify hub/index.html registers the scoped service worker
    hub_index = root / "hub" / "index.html"
    assert hub_index.exists()
    content_hub = hub_index.read_text()
    assert "navigator.serviceWorker.register" in content_hub, "hub/index.html should register its own service worker"
    assert "clintrials/hub/sw.js" in content_hub, "hub/index.html should handle subpath registration"
    assert "clintrials/hub/" in content_hub, "hub/index.html should specify correct subpath scope"

    # 4. Verify hub/sw.js fetch limit and cache name
    hub_sw = root / "hub" / "sw.js"
    assert hub_sw.exists()
    content_sw = hub_sw.read_text()
    assert "CACHE_NAME = 'sim-hub-cache-v5'" in content_sw, "hub/sw.js should use sim-hub-cache-v5"
    assert "!path.startsWith(basePath)" in content_sw, "hub/sw.js fetch event should guard requests"
