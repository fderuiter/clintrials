from __future__ import annotations


class ErrorTemplates:
    # Probability validation template
    PROBABILITY = "{name} must be between 0.0 and 1.0"

    # Positive integer validation template
    POSITIVE_INTEGER = "{name} must be a positive integer"

    # Numeric bound constraint templates
    GE = "{name} must be >= {bound}"
    LE = "{name} must be <= {bound}"
    GT = "{name} must be > {bound}"
    LT = "{name} must be < {bound}"

    # Length validations
    MATCHING_LENGTHS = "{first_name} and {name} should be same length"
    EXPECTED_LENGTH = "{name} should have {expected_length} items"
