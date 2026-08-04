import os
import subprocess
import tempfile


def test_fetch_vendor_soft_fails_and_warns_on_network_failure():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock curl script that always fails
        mock_curl_path = os.path.join(tmpdir, "curl")
        with open(mock_curl_path, "w") as f:
            f.write("#!/bin/bash\nexit 1\n")
        os.chmod(mock_curl_path, 0o755)

        # Setup the environment with the temporary directory at the front of PATH
        env = os.environ.copy()
        env["PATH"] = tmpdir + os.pathsep + env.get("PATH", "")

        # Run fetch_vendor.sh
        result = subprocess.run(
            ["./fetch_vendor.sh"],
            capture_output=True,
            text=True,
            env=env,
            cwd="/app"
        )

        # It must exit with code 0 (soft-fail)
        assert result.returncode == 0

        # Verify it printed the warnings to stderr
        assert "WARNING: Failed to download unreachable dependency:" in result.stderr
        assert "iframeResizer.contentWindow.min.js" in result.stderr
        assert "iframeResizer.min.js" in result.stderr
        assert "nested client-side iframe communication and automatic height resizing" in result.stderr
        assert "interactive embedded frame resizing and layout responsiveness" in result.stderr

        # Since it retries 3 times per download (making 4 attempts total)
        # Verify curl was called 8 times total (4 for each file)
        attempts_count = result.stdout.count("attempt ")
        assert attempts_count == 8

def test_setup_sh_contains_timeouts():
    setup_sh_path = "/app/setup.sh"
    with open(setup_sh_path, "r") as f:
        content = f.read()
    assert "export PIP_TIMEOUT=30" in content
    assert "export PIP_DEFAULT_TIMEOUT=30" in content
    assert "export POETRY_HTTP_TIMEOUT=30" in content

def test_setup_ps1_contains_timeouts():
    setup_ps1_path = "/app/setup.ps1"
    with open(setup_ps1_path, "r") as f:
        content = f.read()
    assert "$env:PIP_TIMEOUT = \"30\"" in content
    assert "$env:PIP_DEFAULT_TIMEOUT = \"30\"" in content
    assert "$env:POETRY_HTTP_TIMEOUT = \"30\"" in content

