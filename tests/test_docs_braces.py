# SPDX-License-Identifier: MIT

import subprocess
from pathlib import Path


def test_escape_mdx_behavior_via_node():
    """
    Test the escapeMdx function in scripts/generate_mdx.js by running a short Node.js script.
    """
    js_code = """
    const fs = require('fs');
    const path = require('path');

    // Read generate_mdx.js and extract escapeMdx
    const mdxScriptPath = path.resolve(process.cwd(), 'scripts/generate_mdx.js');
    const content = fs.readFileSync(mdxScriptPath, 'utf8');

    // Evaluate the file or extract escapeMdx function specifically
    // We can evaluate the function by making a simple sandbox module
    const escapeMdxMatch = content.match(/function escapeMdx\\(str\\) \\{[\\s\\S]*?\\}/);
    if (!escapeMdxMatch) {
        console.error("Could not find escapeMdx function in generate_mdx.js");
        process.exit(1);
    }

    // Create an executable function from the match
    const escapeMdx = new Function('str', escapeMdxMatch[0] + '\\nreturn escapeMdx(str);');

    // Test cases
    const testCases = [
        { input: "hello {world}", expected: "hello {world}" },
        { input: "a | b", expected: "a &#124; b" },
        { input: "{foo} | {bar}", expected: "{foo} &#124; {bar}" },
        { input: "", expected: "" },
    ];

    for (const { input, expected } of testCases) {
        const actual = escapeMdx(input);
        if (actual !== expected) {
            console.error(`Mismatch for input "${input}": expected "${expected}", got "${actual}"`);
            process.exit(1);
        }
    }
    console.log("SUCCESS");
    """

    # Run node subprocess
    project_root = Path(__file__).parent.parent
    result = subprocess.run(
        ["node", "-e", js_code],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Node script failed with stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "SUCCESS" in result.stdout


def test_generated_mdx_braces_preservation():
    """
    Verify that generated mdx and html documents contain raw curly braces and no HTML curly brace entities,
    while still containing escaped pipes.
    """
    project_root = Path(__file__).parent.parent

    # Ensure npm packages are installed so that build scripts can run in clean environments (like Python CI)
    if not (project_root / "node_modules" / "marked").exists():
        subprocess.run(["npm", "install"], cwd=str(project_root), check=True, capture_output=True)

    # Let's ensure build is run so files are up to date
    # Run npm run prebuild && npm run build
    build_result = subprocess.run(
        ["npm", "run", "prebuild"],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    assert build_result.returncode == 0, f"prebuild failed: {build_result.stderr}"

    compile_result = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    assert compile_result.returncode == 0, f"build failed: {compile_result.stderr}"

    reference_dir = project_root / "docs" / "reference"
    dist_dir = project_root / "docs" / "dist"

    # Gather all .mdx and .html files
    mdx_files = list(reference_dir.rglob("*.mdx"))
    html_files = list(dist_dir.rglob("*.html"))

    assert len(mdx_files) > 0, "No MDX files found"
    assert len(html_files) > 0, "No HTML files found"

    # 1. No escaped braces (&#123; or &#125;) in any MDX or HTML file
    for f in mdx_files + html_files:
        content = f.read_text(errors="ignore")
        assert "&#123;" not in content, f"Found escaped open curly brace (&#123;) in {f.relative_to(project_root)}"
        assert "&#125;" not in content, f"Found escaped close curly brace (&#125;) in {f.relative_to(project_root)}"

    # 2. Check that raw curly braces exist in specific files we expect them to be
    # E.g., Random Seed Strategy: {efftox_view_seed_strategy} in efftox_view/index.mdx
    efftox_view_mdx = reference_dir / "clintrials" / "visualization" / "dashboard" / "views" / "efftox_view" / "index.mdx"
    assert efftox_view_mdx.exists()
    content = efftox_view_mdx.read_text()
    assert "{efftox_view_seed_strategy}" in content, "Expected raw curly braces for {efftox_view_seed_strategy}"

    efftox_view_html = dist_dir / "clintrials" / "visualization" / "dashboard" / "views" / "efftox_view" / "index.html"
    assert efftox_view_html.exists()
    html_content = efftox_view_html.read_text()
    assert "{efftox_view_seed_strategy}" in html_content, "Expected raw curly braces for {efftox_view_seed_strategy} in HTML output"

    # 3. Check that escaped pipes (&#124;) exist in some files (like BaseDoseFindingTrial.mdx)
    base_trial_mdx = reference_dir / "clintrials" / "core" / "protocol" / "BaseDoseFindingTrial.mdx"
    if base_trial_mdx.exists():
        trial_content = base_trial_mdx.read_text()
        assert "&#124;" in trial_content, "Expected escaped pipe characters in BaseDoseFindingTrial.mdx"
