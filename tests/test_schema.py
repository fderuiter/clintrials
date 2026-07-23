import pytest


def test_dynamic_bounds_enforcement():  # type: ignore
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
        p_response: Probability = Field(ge=0.2, le=0.8, description="Custom response prob")
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
