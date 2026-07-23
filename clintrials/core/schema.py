from __future__ import annotations

import dataclasses
from typing import Annotated, Any, List, Optional, Union, get_args, get_origin

from clintrials.validation import (
    validate_bounds,
    validate_positive_integer,
    validate_probability,
)


class FieldInfo:
    def __init__(self, default: Any = dataclasses.MISSING, description: Optional[str] = None, ge: Optional[Union[int, float]] = None, le: Optional[Union[int, float]] = None, gt: Optional[Union[int, float]] = None, lt: Optional[Union[int, float]] = None) -> None:
        self.default = default
        self.description = description
        self.ge = ge
        self.le = le
        self.gt = gt
        self.lt = lt
        self.annotation = None

def Field(default: Any = dataclasses.MISSING, description: Optional[str] = None, ge: Optional[Union[int, float]] = None, le: Optional[Union[int, float]] = None, gt: Optional[Union[int, float]] = None, lt: Optional[Union[int, float]] = None, **kwargs: Any) -> Any:
    """Create and return a FieldInfo instance."""
    return FieldInfo(default=default, description=description, ge=ge, le=le, gt=gt, lt=lt)

class BaseModel:
    model_fields: dict[str, Any]
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.model_fields = {}
        import typing
        hints = typing.get_type_hints(cls, include_extras=True)

        for name, ann in hints.items():
            if hasattr(cls, name):
                val = getattr(cls, name)
                if isinstance(val, FieldInfo):
                    val.annotation = ann
                    cls.model_fields[name] = val

                    if val.default is dataclasses.MISSING:
                        setattr(cls, name, dataclasses.field())
                    else:
                        setattr(cls, name, dataclasses.field(default=val.default))
            else:
                f = FieldInfo()
                f.annotation = ann
                cls.model_fields[name] = f

        dataclasses.dataclass(cls)

    def __post_init__(self) -> None:
        for name, f in self.model_fields.items():
            val = getattr(self, name)
            self._validate_value(name, val, f)

    def _validate_value(self, name: str, value: Any, f: Any) -> None:
        if value is None:
            return

        def check_bounds(v: Any, constraints: Any) -> None:
            if constraints.ge is not None or constraints.le is not None:
                validate_bounds(v, lower=constraints.ge, upper=constraints.le, name=name, exclusive=False)
            if constraints.gt is not None or constraints.lt is not None:
                validate_bounds(v, lower=constraints.gt, upper=constraints.lt, name=name, exclusive=True)

        def is_list_annotation(ann: Any) -> bool:
            if ann is None:
                return False
            origin = get_origin(ann)
            if origin is Annotated:
                return is_list_annotation(get_args(ann)[0])
            if origin in (list, tuple):
                return True
            if getattr(ann, "__origin__", None) in (list, tuple):
                return True
            if origin is Union:
                args = get_args(ann)
                for arg in args:
                    if arg is not type(None) and is_list_annotation(arg):
                        return True
            return False

        if is_list_annotation(f.annotation):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"Field '{name}' must be an iterable list rather than a scalar.")

        if isinstance(value, (list, tuple)):
            for item in value:
                check_bounds(item, f)
                self._validate_annotated(name, item, f.annotation)
        else:
            check_bounds(value, f)
            self._validate_annotated(name, value, f.annotation)

    def _validate_annotated(self, name: str, value: Any, annotation: Any) -> None:
        origin = get_origin(annotation)
        if origin is Annotated:
            args = get_args(annotation)
            self._validate_annotated(name, value, args[0])
            is_prob = False
            is_pos_int = False
            for arg in args[1:]:
                if arg == "Probability":
                    is_prob = True
                elif arg == "PositiveInt":
                    is_pos_int = True

                if isinstance(arg, FieldInfo):
                    if is_prob:
                        validate_probability(value, name)
                    if is_pos_int:
                        validate_positive_integer(value, name)

                    if arg.ge is not None or arg.le is not None:
                        validate_bounds(value, lower=arg.ge, upper=arg.le, name=name, exclusive=False)
                    if arg.gt is not None or arg.lt is not None:
                        validate_bounds(value, lower=arg.gt, upper=arg.lt, name=name, exclusive=True)
        elif origin is list or getattr(origin, "__origin__", origin) is list:
            args = get_args(annotation)
            if args:
                self._validate_annotated(name, value, args[0])
        elif origin is type(None) or origin is Union:
            args = get_args(annotation)
            for arg in args:
                if arg is not type(None):
                    self._validate_annotated(name, value, arg)

Probability = Annotated[
    float, "Probability", Field(ge=0.0, le=1.0, description="A valid probability between 0 and 1.")
]
PositiveInt = Annotated[int, "PositiveInt", Field(gt=0, description="A positive integer.")]

class WinRatioSchema(BaseModel):
    num_subjects_A: PositiveInt = Field(
        default=100, description="Number of subjects in Group A"
    )
    num_subjects_B: PositiveInt = Field(
        default=50, description="Number of subjects in Group B"
    )
    num_simulations: PositiveInt = Field(
        default=1000, description="Number of simulations"
    )
    p_y1_A: Probability = Field(
        default=0.50, description="Probability of y1=1 for Group A"
    )
    p_y1_B: Probability = Field(
        default=0.50, description="Probability of y1=1 for Group B"
    )
    p_y2_A: Probability = Field(
        default=0.75, description="Probability of y2=1 for Group A"
    )
    p_y2_B: Probability = Field(
        default=0.25, description="Probability of y2=1 for Group B"
    )
    p_y3_A: Probability = Field(
        default=0.43, description="Probability of y3=1 for Group A"
    )
    p_y3_B: Probability = Field(
        default=0.27, description="Probability of y3=1 for Group B"
    )
    significance_level: Probability = Field(
        default=0.05, description="Significance level"
    )

class CRMSchema(BaseModel):
    prior: List[Probability] = Field(description="Prior probabilities of toxicity")
    target: Probability = Field(description="Target toxicity probability")
    first_dose: PositiveInt = Field(default=1, description="First dose level")
    max_size: PositiveInt = Field(default=30, description="Maximum sample size")
    lowest_dose_too_toxic_hurdle: Probability = Field(
        default=0.0, description="Lowest dose hurdle"
    )
    lowest_dose_too_toxic_certainty: Probability = Field(
        default=0.0, description="Lowest dose certainty"
    )
    coherency_threshold: Probability = Field(
        default=0.0, description="Coherency threshold"
    )
    bootstrap_samples: PositiveInt = Field(default=200, description="Bootstrap samples")
    min_beta: Optional[float] = Field(default=None, description="Minimum beta limit")
    max_beta: Optional[float] = Field(default=None, description="Maximum beta limit")
    n_points: Optional[PositiveInt] = Field(default=None, description="Integration point count")
    sample_size: Optional[PositiveInt] = Field(default=None, description="Monte Carlo sample size")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_beta is not None and self.max_beta is not None:
            if self.min_beta >= self.max_beta:
                raise ValueError("min_beta must be less than max_beta")

class EffToxSchema(BaseModel):
    real_doses: List[float] = Field(description="Real dose values")
    prior_tox_probs: Optional[List[Probability]] = Field(
        default=None, description="Prior tox probs"
    )
    prior_eff_probs: Optional[List[Probability]] = Field(
        default=None, description="Prior eff probs"
    )
    tox_cutoff: Optional[Probability] = Field(
        default=None, description="Toxicity cutoff"
    )
    eff_cutoff: Optional[Probability] = Field(
        default=None, description="Efficacy cutoff"
    )
    tox_certainty: Optional[Probability] = Field(
        default=None, description="Toxicity certainty"
    )
    eff_certainty: Optional[Probability] = Field(
        default=None, description="Efficacy certainty"
    )
    max_size: Optional[PositiveInt] = Field(default=None, description="Maximum size")
    first_dose: PositiveInt = Field(default=1, description="First dose level")

