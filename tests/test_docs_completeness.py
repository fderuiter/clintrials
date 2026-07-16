import re
from pathlib import Path

def test_all_public_modules_documented():
    # 1. Read documented modules from index.rst
    index_path = Path("docs/reference/index.rst")
    with open(index_path, "r") as f:
        content = f.read()
    
    documented_modules = set()
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("clintrials."):
            documented_modules.add(line)

    # 2. Discover all public Python files on disk
    # Exclude internal helper files, test directories, and private Python modules
    src_dir = Path("clintrials")
    public_modules_on_disk = set()
    
    # helper file names to exclude
    exclude_names = {"utils.py", "helpers.py", "validation.py", "errors.py"}
    
    for py_file in src_dir.rglob("*.py"):
        # Exclude private modules
        if py_file.name.startswith("_"):
            continue
            
        # Exclude internal helper files
        if py_file.name in exclude_names:
            continue
            
        # Convert path to module string
        # e.g. clintrials/core/math.py -> clintrials.core.math
        module_str = str(py_file.with_suffix("")).replace("/", ".")
        public_modules_on_disk.add(module_str)

    # 3. Check for missing modules
    missing_modules = public_modules_on_disk - documented_modules
    
    assert not missing_modules, (
        f"Found {len(missing_modules)} public modules on disk that are missing "
        f"from the manual documentation index ({index_path}):\n"
        + "\n".join(sorted(missing_modules))
        + "\n\nPlease add these modules to the reference index."
    )
