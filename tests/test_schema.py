# SPDX-License-Identifier: MIT

import pytest


def test_dynamic_bounds_enforcement():
    from typing import Annotated

    from clintrials.core.schema import BaseModel, Field, PositiveInt, Probability

    # 1. Standard probability remains operational
    class StandardSchema(BaseModel):
        prob: Probability = Field(description="A standard probability")
        pos_int: PositiveInt = Field(description="A standard positive integer")

    # Should not raise for valid standard values
    StandardSchema(prob=0.5, pos_int=10)  # type: ignore[call-arg]

    # Should raise for standard boundaries
    with pytest.raises(ValueError, match="prob must be between 0.0 and 1.0"):
        StandardSchema(prob=1.5, pos_int=10)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="pos_int must be a positive integer"):
        StandardSchema(prob=0.5, pos_int=0)  # type: ignore[call-arg]

    # 2. Customized boundaries on the Field itself
    class CustomFieldSchema(BaseModel):
        p_response: Probability = Field(
            ge=0.2, le=0.8, description="Custom response prob"
        )
        size: PositiveInt = Field(gt=5, lt=20, description="Custom size limit")

    # Should not raise for values strictly within the customized range
    CustomFieldSchema(p_response=0.5, size=10)  # type: ignore[call-arg]

    # Should raise precise error messages with valid ranges when violated
    with pytest.raises(ValueError, match="p_response must be <= 0.8"):
        CustomFieldSchema(p_response=0.9, size=10)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="p_response must be >= 0.2"):
        CustomFieldSchema(p_response=0.1, size=10)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="size must be > 5"):
        CustomFieldSchema(p_response=0.5, size=5)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="size must be < 20"):
        CustomFieldSchema(p_response=0.5, size=20)  # type: ignore[call-arg]

    # 3. Customized boundaries on custom Annotated types
    CustomProbType = Annotated[float, "Probability", Field(ge=0.3, le=0.7)]
    CustomIntType = Annotated[int, "PositiveInt", Field(gt=10, lt=50)]

    class CustomAnnotatedTypeSchema(BaseModel):
        p_custom: CustomProbType
        count_custom: CustomIntType

    # Should not raise for valid values
    CustomAnnotatedTypeSchema(p_custom=0.5, count_custom=30)  # type: ignore[call-arg]

    # Should raise precise error messages for annotated type boundary violations
    with pytest.raises(ValueError, match="p_custom must be <= 0.7"):
        CustomAnnotatedTypeSchema(p_custom=0.8, count_custom=30)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="p_custom must be >= 0.3"):
        CustomAnnotatedTypeSchema(p_custom=0.2, count_custom=30)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="count_custom must be > 10"):
        CustomAnnotatedTypeSchema(p_custom=0.5, count_custom=10)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="count_custom must be < 50"):
        CustomAnnotatedTypeSchema(p_custom=0.5, count_custom=50)  # type: ignore[call-arg]


def test_version_schema_enforcement():
    from clintrials.core.schema import BaseModel, Field, Version

    class InlineVersionSchema(BaseModel):
        version: Version = Field(description="Package or hub version")

    # Valid PEP 440 versions should be accepted
    InlineVersionSchema(version="1.0.0")  # type: ignore[call-arg]
    InlineVersionSchema(version="2.3.4.dev1")  # type: ignore[call-arg]
    InlineVersionSchema(version="1.0a1")  # type: ignore[call-arg]

    # Non-PEP 440 versions (like 'latest') should be rejected with the standard error
    with pytest.raises(
        ValueError, match="version must be a valid PEP 440 version string"
    ):
        InlineVersionSchema(version="latest")  # type: ignore[call-arg]

    with pytest.raises(
        ValueError, match="version must be a valid PEP 440 version string"
    ):
        InlineVersionSchema(version="invalid-version")  # type: ignore[call-arg]

    with pytest.raises(
        ValueError, match="version must be a valid PEP 440 version string"
    ):
        InlineVersionSchema(version="")  # type: ignore[call-arg]


def test_schema_serialization():
    from clintrials.core.schema import (
        BaseModel,
        WagesTaitSchema,
    )
    from scripts.serialize_schemas import generate_schema_for_class

    subclasses = BaseModel.__subclasses__()
    assert len(subclasses) >= 6
    subclass_names = [cls.__name__ for cls in subclasses]
    for expected in [
        "WinRatioSchema",
        "CRMSchema",
        "EffToxSchema",
        "WagesTaitSchema",
        "WATUSchema",
        "GroupSequentialDesignSchema",
    ]:
        assert expected in subclass_names

    wages_tait_schema = generate_schema_for_class(WagesTaitSchema)
    assert wages_tait_schema["type"] == "object"
    assert "skeletons" in wages_tait_schema["properties"]
    assert wages_tait_schema["properties"]["skeletons"]["type"] == "array"
    assert wages_tait_schema["properties"]["skeletons"]["items"]["type"] == "array"


def test_worker_side_validation_guard():
    from hub.runner import validate_fields

    # 1. Valid CRMSchema payload
    valid_crm = {
        "prior": [0.1, 0.2, 0.3],
        "target": 0.25,
        "first_dose": 1,
        "max_size": 30,
    }
    errors = validate_fields("CRMSchema", valid_crm)
    assert not errors

    # 2. Invalid CRMSchema payload (out of bounds)
    invalid_crm = {
        "prior": [0.1, -0.2, 0.3],  # negative probability
        "target": 1.5,  # target > 1.0
        "first_dose": 1,
        "max_size": 30,
    }
    errors = validate_fields("CRMSchema", invalid_crm)
    assert "prior" in errors
    assert "target" in errors

    # 3. Invalid GroupSequentialDesignSchema (alpha >= 1.0)
    invalid_gsd = {
        "k": 3,
        "alpha": 1.0,
        "timing": [0.33, 0.67, 1.0],
        "n_sims": 1000,
        "theta": 1.0,
    }
    errors = validate_fields("GroupSequentialDesignSchema", invalid_gsd)
    assert "alpha" in errors
    assert "alpha must be between 0.0 and 1.0" in errors["alpha"]
