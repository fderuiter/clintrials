import os
import subprocess
import sys
import time
import urllib.request

import pytest
from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import Page

# Define viewports to test
VIEWPORTS = [
    {"width": 375, "height": 667},  # Mobile
    {"width": 1280, "height": 720}, # Desktop
]

@pytest.fixture(scope="module")
def streamlit_server():  # type: ignore
    # Start the Streamlit app
    env = os.environ.copy()

    process = subprocess.Popen(
        [sys.executable, "-m", "clintrials.visualization.dashboard.launcher", "--port", "8502", "--server.headless=true"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for the server to start
    url = "http://localhost:8502"
    for _ in range(30):
        try:
            response = urllib.request.urlopen(url)
            if response.status == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        process.terminate()
        raise RuntimeError("Streamlit server did not start in time.")

    yield url

    # Teardown
    process.terminate()
    process.wait()

@pytest.fixture(params=VIEWPORTS, ids=["mobile", "desktop"])
def viewport(request):  # type: ignore
    return request.param

def test_dashboard_accessibility(page: Page, streamlit_server: str, viewport: dict, tmp_path):  # type: ignore
    # Set viewport
    page.set_viewport_size(viewport)  # type: ignore

    # Navigate to the dashboard
    page.goto(streamlit_server)

    # Wait for the main Streamlit container to load
    page.wait_for_selector(".stApp")

    # Wait for the table to appear by checking for the summary text
    # Since Preview Mode is the default, the dashboard automatically runs a simulation and displays results.
    try:
        page.wait_for_selector("text=Simulation Summary", timeout=30000)
    except Exception as e:
        with open(f"debug_html_error_{viewport['width']}.html", "w") as f:
            f.write(page.content())
        raise e

    # Wait a bit longer for actual table rendering
    page.wait_for_timeout(2000)

    import json
    import os

    # Run Axe audit for standard mode
    axe = Axe()
    results_standard = axe.run(page)

    # Strip HTML to prevent PII leakage and filter violations
    def process_violations(violations):  # type: ignore
        processed = []
        for v in violations:
            for node in v["nodes"]:
                node.pop("html", None)
            processed.append(v)
        return processed

    violations_standard = process_violations(results_standard.response["violations"])  # type: ignore

    # Save standard mode report
    os.makedirs("accessibility_reports", exist_ok=True)
    with open(f"accessibility_reports/axe_standard_{viewport['width']}x{viewport['height']}.json", "w") as f:
        json.dump(violations_standard, f, indent=2)

    # Check specifically for ARIA table elements (Requirement 2 & 3)
    assert page.locator("thead").count() > 0, "No <thead> found in the rendered tables."
    assert page.locator("th").count() > 0, "No <th> found in the rendered tables."

    assert len(violations_standard) == 0, f"Standard Mode Accessibility Violations: {violations_standard}"
