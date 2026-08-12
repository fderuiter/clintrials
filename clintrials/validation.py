# SPDX-License-Identifier: MIT

"""Validation functions for the Clinical Trials library."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from packaging.version import InvalidVersion, Version

from clintrials.core.errors import ErrorTemplates


def validate_matching_lengths(**kwargs: Sequence[Any]) -> None:
    """Validates that all provided arrays have the same length.

    Pass arrays as keyword arguments. For example:
    validate_matching_lengths(array1=arr1, array2=arr2)

    Args:
        **kwargs: Arbitrary keyword arguments where keys are the names
            of the arrays and values are the arrays themselves.

    Returns:
        None

    Raises:
        ValueError: If any array does not match the length of the first array.
    """
    if not kwargs:
        return

    iterator = iter(kwargs.items())
    first_name, first_arr = next(iterator)
    expected_len = len(first_arr)

    for name, arr in iterator:
        if len(arr) != expected_len:
            raise ValueError(
                ErrorTemplates.MATCHING_LENGTHS.format(first_name=first_name, name=name)
            )


def validate_expected_length(
    array: Sequence[Any], expected_length: int, name: str
) -> None:
    """Validates that an array has exactly the expected length.

    Args:
        array (list or numpy.ndarray): The array to validate.
        expected_length (int): The expected length of the array.
        name (str): The name of the parameter being validated, used in the error message.

    Returns:
        None

    Raises:
        ValueError: If the array length does not match the expected length.
    """
    if len(array) != expected_length:
        raise ValueError(
            ErrorTemplates.EXPECTED_LENGTH.format(
                name=name, expected_length=expected_length
            )
        )


def validate_bounds(
    value: Union[float, int],
    lower: Optional[Union[float, int]],
    upper: Optional[Union[float, int]],
    name: str,
    exclusive: bool = False,
) -> None:
    """Validates that a numerical value is within the specified bounds.

    Args:
        value (float or int): The numerical value to validate.
        lower (float or int, optional): The lower bound.
        upper (float or int, optional): The upper bound.
        name (str): The name of the parameter, used in the error message.
        exclusive (bool, optional): If True, bounds are exclusive (value > lower and value < upper).
            If False, bounds are inclusive (value >= lower and value <= upper). Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If the value is outside the specified bounds.
    """
    if exclusive:
        if lower is not None and value <= lower:
            raise ValueError(ErrorTemplates.GT.format(name=name, bound=lower))
        if upper is not None and value >= upper:
            raise ValueError(ErrorTemplates.LT.format(name=name, bound=upper))
    else:
        if lower is not None and value < lower:
            raise ValueError(ErrorTemplates.GE.format(name=name, bound=lower))
        if upper is not None and value > upper:
            raise ValueError(ErrorTemplates.LE.format(name=name, bound=upper))


def validate_probability(value: float, name: str, exclusive: bool = False) -> None:
    """Validates that a value is a valid probability between 0 and 1.

    Args:
        value (float): The probability value to validate.
        name (str): The name of the parameter, used in the error message.
        exclusive (bool, optional): If True, probabilities of exactly 0 or 1 are invalid.
            Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If the value is not a valid probability.
    """
    if exclusive:
        if not (0 < value < 1):
            raise ValueError(ErrorTemplates.PROBABILITY.format(name=name))
    else:
        if not (0 <= value <= 1):
            raise ValueError(ErrorTemplates.PROBABILITY.format(name=name))


def validate_positive_integer(value: int, name: str) -> None:
    """Validates that a value is a positive integer.

    Args:
        value (int): The value to validate.
        name (str): The name of the parameter, used in the error message.

    Returns:
        None

    Raises:
        ValueError: If the value is not an integer or is less than or equal to zero.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(ErrorTemplates.POSITIVE_INTEGER.format(name=name))


def validate_version(value: Any, name: str) -> None:
    """Validates that a value is a valid PEP 440 version string.

    Args:
        value (Any): The value to validate.
        name (str): The name of the parameter, used in the error message.

    Returns:
        None

    Raises:
        ValueError: If the value is not a string, is an empty string, or is not a
            valid PEP 440 version string.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(ErrorTemplates.PEP440_VERSION.format(name=name))

    try:
        Version(value)
    except InvalidVersion as e:
        raise ValueError(ErrorTemplates.PEP440_VERSION.format(name=name)) from e


def _get_defined_personas_from_strategy() -> list[str]:
    from pathlib import Path
    strategy_path = Path("/app/PRODUCT_STRATEGY.md")
    if not strategy_path.exists():
        strategy_path = Path("PRODUCT_STRATEGY.md")
    if not strategy_path.exists():
        return ["dr. aris thorne", "eleanor vance", "biostatistician", "data scientist", "developer"]

    content = strategy_path.read_text()
    possible_personas = ["dr. aris thorne", "eleanor vance", "biostatistician", "data scientist", "developer"]
    defined = []
    content_lower = content.lower()
    for persona in possible_personas:
        if persona in content_lower:
            defined.append(persona)
        elif persona == "dr. aris thorne" and "aris thorne" in content_lower:
            defined.append(persona)
    return defined


def _is_persona_referenced(input_text: str, defined_personas: list[str]) -> bool:
    input_lower = input_text.lower()
    for persona in defined_personas:
        if persona == "dr. aris thorne":
            if any(term in input_lower for term in ["dr. aris thorne", "aris thorne", "thorne", "dr. thorne"]):
                return True
        elif persona == "eleanor vance":
            if any(term in input_lower for term in ["eleanor vance", "eleanor", "vance"]):
                return True
        else:
            if persona in input_lower:
                return True
            if persona.rstrip("s") in input_lower:
                return True
    return False


def _get_roadmap_milestones() -> list[str]:
    from pathlib import Path
    roadmap_path = Path("/app/ROADMAP.md")
    if not roadmap_path.exists():
        roadmap_path = Path("ROADMAP.md")
    if not roadmap_path.exists():
        return []

    content = roadmap_path.read_text()
    milestones = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("-") or line.startswith("*"):
            milestone_text = line.lstrip("-* ").strip()
            if milestone_text:
                milestones.append(milestone_text)
    return milestones


def validate_feature_request(issue_data: dict[str, Any]) -> bool:
    """Validates a feature request submission based on dual-track rules.

    The unified Technical / Infrastructure Track requires a roadmap milestone reference.
    The unified User-Centric / Clinical Track bypasses the roadmap milestone requirement,
    relying instead on alignment with PRODUCT_STRATEGY.md without requiring
    low-level code or technical metrics.

    Args:
        issue_data (dict): A dictionary representing the submitted issue.
            Expected keys: 'track', 'roadmap_milestone', 'clinical_pillar',
            'user_persona', 'solution_description'.

    Returns:
        bool: True if the issue data is valid.

    Raises:
        ValueError: If validation fails.
    """
    track = str(issue_data.get("track", "")).lower()

    if "user-centric" in track or "clinical" in track:
        clinical_pillar = issue_data.get("clinical_pillar") or issue_data.get("user_persona") or issue_data.get("persona")
        if not clinical_pillar or str(clinical_pillar).strip() in ("", "None", "N/A"):
            raise ValueError(
                "User-Centric / Clinical track requires a reference to PRODUCT_STRATEGY.md strategic pillars or personas."
            )

        defined_personas = _get_defined_personas_from_strategy()
        if not _is_persona_referenced(str(clinical_pillar), defined_personas):
            raise ValueError(
                "User-Centric / Clinical track proposals must reference a defined strategic persona (e.g., Dr. Aris Thorne, Eleanor Vance, biostatistician, or data scientist)."
            )
        return True

    elif "infrastructure" in track or "technical" in track:
        roadmap_milestone = issue_data.get("roadmap_milestone")
        if not roadmap_milestone or str(roadmap_milestone).strip() in ("", "None", "N/A"):
            raise ValueError(
                "Technical / Infrastructure track requires a valid ROADMAP.md milestone reference."
            )

        active_milestones = _get_roadmap_milestones()
        norm_ref = str(roadmap_milestone).strip().lower()

        import re
        def clean_str(s: str) -> str:
            return re.sub(r'[^a-z0-9]', '', s.lower())

        norm_ref_clean = clean_str(norm_ref)

        matched = False
        for am in active_milestones:
            am_clean = clean_str(am)
            if norm_ref_clean in am_clean or am_clean in norm_ref_clean:
                matched = True
                break

            if len(norm_ref) >= 15:
                for i in range(len(norm_ref) - 14):
                    sub = norm_ref[i:i+15]
                    if sub in am.lower():
                        matched = True
                        break
            if matched:
                break

        if not matched:
            raise ValueError(
                "Technical / Infrastructure track requires a valid and active ROADMAP.md milestone reference."
            )
        return True

    else:
        raise ValueError(
            "Invalid track selection. Please select either 'User-Centric / Clinical Track' or 'Technical / Infrastructure Track'."
        )
