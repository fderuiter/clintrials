"""Tests for clintrials/core/errors.py."""


def test_error_templates_pep440_format() -> None:
    """Verify that PEP440_VERSION error template formats correctly."""
    from clintrials.core.errors import ErrorTemplates

    msg = ErrorTemplates.PEP440_VERSION.format(name="my_version")
    assert msg == "my_version must be a valid PEP 440 version string"


def test_all_error_templates_have_name() -> None:
    """Verify that all standard error templates have {name} for formatting."""
    from clintrials.core.errors import ErrorTemplates

    templates = [
        ErrorTemplates.PROBABILITY,
        ErrorTemplates.POSITIVE_INTEGER,
        ErrorTemplates.GE,
        ErrorTemplates.LE,
        ErrorTemplates.GT,
        ErrorTemplates.LT,
        ErrorTemplates.EXPECTED_LENGTH,
        ErrorTemplates.PEP440_VERSION,
    ]
    for template in templates:
        assert "{name}" in template
