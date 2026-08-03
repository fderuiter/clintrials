import re
from pathlib import Path


def test_hub_service_worker_inline_registration() -> None:
    """Verify that the Simulation Hub entry page (hub/index.html) has a valid,

    ES5-compatible inline service worker registration script inside its <head>
    that correctly resolves both standard/root and subpath scopes.
    """
    root = Path(__file__).parent.parent
    hub_index = root / "hub" / "index.html"

    assert hub_index.exists(), "hub/index.html does not exist!"

    content = hub_index.read_text(encoding="utf-8")

    # Check that <head> and </head> exist
    assert "<head>" in content
    assert "</head>" in content

    # Locate everything inside <head>
    head_content = content.split("<head>")[1].split("</head>")[0]

    # Verify there is an inline service worker script
    assert "navigator.serviceWorker.register" in head_content, "Service worker registration not found inside <head>."

    # Check ES5 compatibility inside the script tag containing the registration
    script_match = re.search(
        r"<script>.*?(navigator\.serviceWorker\.register).*?</script>",
        head_content,
        re.DOTALL
    )
    assert script_match is not None, "Failed to locate the inline script block for service worker registration."

    script_block = script_match.group(0)

    # Verify ES5 compatibility:
    # No "const" variable declarations (use "var" instead)
    assert "const " not in script_block, "Found 'const' in inline script, violating ES5 legacy compatibility."
    # No "let" variable declarations (use "var" instead)
    assert "let " not in script_block, "Found 'let' in inline script, violating ES5 legacy compatibility."
    # No ES6 arrow functions (use "function" instead)
    assert "=>" not in script_block, "Found '=>' (arrow function) in inline script, violating ES5 legacy compatibility."
    # No ES6 string/array includes() (use indexOf() !== -1 instead)
    assert ".includes(" not in script_block, "Found '.includes(' in inline script, violating ES5 legacy compatibility."

    # Verify that standard root scope and subpath scope are handled correctly
    assert "/clintrials/" in script_block, "Subpath scope '/clintrials/' is not configured/handled in registration."
    assert "indexOf" in script_block, "indexOf should be used to check path for subpath."
    assert "swUrl" in script_block, "swUrl variable should be defined."
    assert "swScope" in script_block, "swScope variable should be defined."
