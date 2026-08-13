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
            f.write("#!/bin/bash\necho \"poetry called with: $@\"\nexit 0\n")
        os.chmod(poetry_path, 0o755)

        # Build custom PATH that includes our tmpdir but does NOT include any real pandoc
        # We do this by creating a directory of symlinks to standard system commands, excluding pandoc.
        env = os.environ.copy()
        bin_dir = os.path.join(tmpdir, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        for path_dir in env.get("PATH", "").split(os.pathsep):
            if not path_dir or not os.path.exists(path_dir):
                continue
            if "pandoc" in path_dir.lower():
                continue
            try:
                for entry in os.scandir(path_dir):
                    if entry.is_file() and os.access(entry.path, os.X_OK):
                        if entry.name != "pandoc":
                            link_path = os.path.join(bin_dir, entry.name)
                            if not os.path.exists(link_path):
                                os.symlink(entry.path, link_path)
            except Exception:
                pass
        env["PATH"] = tmpdir + os.pathsep + bin_dir

        # Run setup.sh
        result = subprocess.run(
            ["./setup.sh"],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )

        assert result.returncode == 0
        assert "WARNING: pandoc is not installed. Documentation testing is skipped." in result.stdout
        assert "To install pandoc, please refer to: https://pandoc.org/installing.html" in result.stdout
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
            f.write("#!/bin/bash\necho \"poetry called with: $@\"\nexit 0\n")
        os.chmod(poetry_path, 0o755)

        # Create mock pandoc
        pandoc_path = os.path.join(tmpdir, "pandoc")
        with open(pandoc_path, "w") as f:
            f.write("#!/bin/bash\necho \"mock pandoc called\"\nexit 0\n")
        os.chmod(pandoc_path, 0o755)

        env = os.environ.copy()
        env["PATH"] = tmpdir + os.pathsep + env.get("PATH", "")

        # Run setup.sh
        result = subprocess.run(
            ["./setup.sh"],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )

        assert result.returncode == 0
        assert "WARNING: pandoc is not installed" not in result.stdout
        assert "poetry called with: run make -C docs doctest" in result.stdout
