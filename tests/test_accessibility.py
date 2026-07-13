import subprocess
import time
import pytest
import os
import sys
import urllib.request
from playwright.sync_api import Page
from axe_playwright_python.sync_playwright import Axe

# Define viewports to test
VIEWPORTS = [
    {"width": 375, "height": 667},  # Mobile
    {"width": 1280, "height": 720}, # Desktop
]

@pytest.fixture(scope="module")
def streamlit_server():
    # Start the Streamlit app
    env = os.environ.copy()
    
    process = subprocess.Popen(
        [sys.executable, "-m", "clintrials.visualization.dashboard.launcher", "--port", "8502", "--server.headless=true"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
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
def viewport(request):
    return request.param

def test_dashboard_accessibility(page: Page, streamlit_server: str, viewport: dict, tmp_path):
    # Set viewport
    page.set_viewport_size(viewport)
    
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

    # Define elements to exclude from structural audits (Streamlit internal components)
    axe_context = {
        "exclude": [
            [".stSidebar"],
            ["[data-testid='stFileUploader']"],
            [".stJson"],
            ["[data-testid='stAlert']"],
            ["[data-testid='stFileUploaderDropzoneInput']"],
            [".stFileChip"],
            ["div[aria-haspopup='true']"]
        ]
    }

    # Run Axe audit for standard mode
    axe = Axe()
    results_standard = axe.run(page, context=axe_context)
    
    # Strip HTML to prevent PII leakage and filter violations
    def process_violations(violations):
        processed = []
        for v in violations:
            if v["impact"] in ["critical", "serious"]:
                for node in v["nodes"]:
                    node.pop("html", None)
                processed.append(v)
        return processed
        
    violations_standard = process_violations(results_standard.response["violations"])
    
    # Save standard mode report
    os.makedirs("accessibility_reports", exist_ok=True)
    with open(f"accessibility_reports/axe_standard_{viewport['width']}x{viewport['height']}.json", "w") as f:
        json.dump(violations_standard, f, indent=2)
    
    # Check specifically for ARIA table elements (Requirement 2 & 3)
    assert page.locator("thead").count() > 0, "No <thead> found in the rendered tables."
    assert page.locator("th").count() > 0, "No <th> found in the rendered tables."
    
    assert len(violations_standard) == 0, f"Standard Mode Accessibility Violations: {violations_standard}"
    
    # Toggle 'Accessibility Mode'
    page.locator("text=Accessibility Mode").evaluate("node => node.closest('label').querySelector('input').click()")
    page.wait_for_timeout(3000)
    
    # Run Axe audit for high-contrast mode
    results_hc = axe.run(page, context=axe_context)
    violations_hc = process_violations(results_hc.response["violations"])
    
    with open(f"accessibility_reports/axe_hc_{viewport['width']}x{viewport['height']}.json", "w") as f:
        json.dump(violations_hc, f, indent=2)
        
    assert len(violations_hc) == 0, f"High-Contrast Mode Accessibility Violations: {violations_hc}"
    
    # Confirm application of the high-contrast palette [cite:source1][cite:source2]
    # Check iframe contents for Plotly colors
    found_hc_color = False
    for frame in page.frames:
        content_lower = frame.content().lower()
        if "#8a5f00" in content_lower or "rgb(138, 95, 0)" in content_lower:
            found_hc_color = True
            break
            
    assert found_hc_color, "High-contrast palette color (#8A5F00) was not found in the rendered charts."
