# SPDX-License-Identifier: MIT

import os
import shutil
import subprocess
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_setup_sh_without_pandoc():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy setup.sh to temporary directory
        setup_sh_src = os.path.join(PROJECT_ROOT, "setup.sh")
        setup_sh_dest = os.path.join(tmpdir, "setup.sh")
        shutil.copy(setup_sh_src, setup_sh_dest)
        os.chmod(setup_sh_dest, 0o755)

        # Create a mock fetch_vendor.sh
        fetch_vendor_path = os.path.join(tmpdir, "fetch_vendor.sh")
        with open(fetch_vendor_path, "w") as f:
            f.write("#!/bin/bash\necho 'mock fetch_vendor.sh called'\nexit 0\n")
        os.chmod(fetch_vendor_path, 0o755)

        # Create mock poetry
        poetry_path = os.path.join(tmpdir, "poetry")
        with open(poetry_path, "w") as f:
            f.write('#!/bin/bash\necho "poetry called with: $@"\nexit 0\n')
        os.chmod(poetry_path, 0o755)

        # Set PATH to ONLY contain tmpdir to completely isolate the environment and guarantee pandoc is absent
        env = os.environ.copy()
        env["PATH"] = tmpdir

        # Run setup.sh
        bash_executable = shutil.which("bash") or "bash"
        result = subprocess.run(
            [bash_executable, "./setup.sh"],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )

        assert result.returncode == 0
        assert (
            "WARNING: pandoc is not installed. Documentation testing is skipped."
            in result.stdout
        )
        assert (
            "To install pandoc, please refer to: https://pandoc.org/installing.html"
            in result.stdout
        )
        assert "poetry called with: run make -C docs doctest" not in result.stdout


def test_setup_sh_with_pandoc():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy setup.sh to temporary directory
        setup_sh_src = os.path.join(PROJECT_ROOT, "setup.sh")
        setup_sh_dest = os.path.join(tmpdir, "setup.sh")
        shutil.copy(setup_sh_src, setup_sh_dest)
        os.chmod(setup_sh_dest, 0o755)

        # Create a mock fetch_vendor.sh
        fetch_vendor_path = os.path.join(tmpdir, "fetch_vendor.sh")
        with open(fetch_vendor_path, "w") as f:
            f.write("#!/bin/bash\necho 'mock fetch_vendor.sh called'\nexit 0\n")
        os.chmod(fetch_vendor_path, 0o755)

        # Create mock poetry
        poetry_path = os.path.join(tmpdir, "poetry")
        with open(poetry_path, "w") as f:
            f.write('#!/bin/bash\necho "poetry called with: $@"\nexit 0\n')
        os.chmod(poetry_path, 0o755)

        # Create mock pandoc
        pandoc_path = os.path.join(tmpdir, "pandoc")
        with open(pandoc_path, "w") as f:
            f.write('#!/bin/bash\necho "mock pandoc called"\nexit 0\n')
        os.chmod(pandoc_path, 0o755)

        env = os.environ.copy()
        env["PATH"] = tmpdir + os.pathsep + env.get("PATH", "")

        # Run setup.sh
        bash_executable = shutil.which("bash") or "bash"
        result = subprocess.run(
            [bash_executable, "./setup.sh"],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )

        assert result.returncode == 0
        assert "WARNING: pandoc is not installed" not in result.stdout
        assert "poetry called with: run make -C docs doctest" in result.stdout
