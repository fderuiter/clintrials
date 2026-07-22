from __future__ import annotations

import dataclasses
from typing import Annotated, List, Optional, Union, get_args, get_origin

from clintrials.validation import (
    validate_bounds,
    validate_positive_integer,
    validate_probability,
)


class FieldInfo:
    def __init__(self, default=dataclasses.MISSING, description=None, ge=None, le=None, gt=None, lt=None):  # type: ignore
        self.default = default
        self.description = description
        self.ge = ge
        self.le = le
        self.gt = gt
        self.lt = lt
        self.annotation = None

def Field(default=dataclasses.MISSING, description=None, ge=None, le=None, gt=None, lt=None, **kwargs):  # type: ignore
    """Create and return a FieldInfo instance."""
    return FieldInfo(default=default, description=description, ge=ge, le=le, gt=gt, lt=lt)  # type: ignore

class BaseModel:
    def __init_subclass__(cls, **kwargs):  # type: ignore
        super().__init_subclass__(**kwargs)
        cls.model_fields = {}  # type: ignore
        import typing
        hints = typing.get_type_hints(cls, include_extras=True)

        for name, ann in hints.items():
            if hasattr(cls, name):
                val = getattr(cls, name)
                if isinstance(val, FieldInfo):
                    val.annotation = ann
                    cls.model_fields[name] = val  # type: ignore

                    if val.default is dataclasses.MISSING:
                        setattr(cls, name, dataclasses.field())
                    else:
                        setattr(cls, name, dataclasses.field(default=val.default))
            else:
                f = FieldInfo()  # type: ignore
                f.annotation = ann
                cls.model_fields[name] = f  # type: ignore

        dataclasses.dataclass(cls)

    def __post_init__(self):  # type: ignore
        for name, f in self.model_fields.items():  # type: ignore
            val = getattr(self, name)
            self._validate_value(name, val, f)  # type: ignore

    def _validate_value(self, name, value, f):  # type: ignore
        if value is None:
            return

        def check_bounds(v, constraints):  # type: ignore
            if constraints.ge is not None or constraints.le is not None:
                validate_bounds(v, lower=constraints.ge, upper=constraints.le, name=name, exclusive=False)
            if constraints.gt is not None or constraints.lt is not None:
                validate_bounds(v, lower=constraints.gt, upper=constraints.lt, name=name, exclusive=True)

        def is_list_annotation(ann):  # type: ignore
            if ann is None:
                return False
            origin = get_origin(ann)
            if origin is Annotated:
                return is_list_annotation(get_args(ann)[0])  # type: ignore
            if origin in (list, tuple):
                return True
            if getattr(ann, "__origin__", None) in (list, tuple):
                return True
            if origin is Union:
                args = get_args(ann)
                for arg in args:
                    if arg is not type(None) and is_list_annotation(arg):  # type: ignore
                        return True
            return False

        if is_list_annotation(f.annotation):  # type: ignore
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"Field '{name}' must be an iterable list rather than a scalar.")

        if isinstance(value, (list, tuple)):
            for item in value:
                check_bounds(item, f)  # type: ignore
                self._validate_annotated(name, item, f.annotation)  # type: ignore
        else:
            check_bounds(value, f)  # type: ignore
            self._validate_annotated(name, value, f.annotation)  # type: ignore

    def _validate_annotated(self, name, value, annotation):  # type: ignore
        origin = get_origin(annotation)
        if origin is Annotated:
            args = get_args(annotation)
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
                    elif is_pos_int:
                        validate_positive_integer(value, name)
                    else:
                        if arg.ge is not None or arg.le is not None:
                            validate_bounds(value, lower=arg.ge, upper=arg.le, name=name, exclusive=False)
                        if arg.gt is not None or arg.lt is not None:
                            validate_bounds(value, lower=arg.gt, upper=arg.lt, name=name, exclusive=True)
        elif origin is list or getattr(origin, "__origin__", origin) is list:
            args = get_args(annotation)
            if args:
                self._validate_annotated(name, value, args[0])  # type: ignore
        elif origin is type(None) or origin is Union:
            args = get_args(annotation)
            for arg in args:
                if arg is not type(None):
                    self._validate_annotated(name, value, arg)  # type: ignore

Probability = Annotated[
    float, "Probability", Field(ge=0.0, le=1.0, description="A valid probability between 0 and 1.")
]
PositiveInt = Annotated[int, "PositiveInt", Field(gt=0, description="A positive integer.")]

class WinRatioSchema(BaseModel):  # type: ignore
    num_subjects_A: PositiveInt = Field(  # type: ignore
        default=100, description="Number of subjects in Group A"
    )
    num_subjects_B: PositiveInt = Field(  # type: ignore
        default=50, description="Number of subjects in Group B"
    )
    num_simulations: PositiveInt = Field(  # type: ignore
        default=1000, description="Number of simulations"
    )
    p_y1_A: Probability = Field(  # type: ignore
        default=0.50, description="Probability of y1=1 for Group A"
    )
    p_y1_B: Probability = Field(  # type: ignore
        default=0.50, description="Probability of y1=1 for Group B"
    )
    p_y2_A: Probability = Field(  # type: ignore
        default=0.75, description="Probability of y2=1 for Group A"
    )
    p_y2_B: Probability = Field(  # type: ignore
        default=0.25, description="Probability of y2=1 for Group B"
    )
    p_y3_A: Probability = Field(  # type: ignore
        default=0.43, description="Probability of y3=1 for Group A"
    )
    p_y3_B: Probability = Field(  # type: ignore
        default=0.27, description="Probability of y3=1 for Group B"
    )
    significance_level: Probability = Field(  # type: ignore
        default=0.05, description="Significance level"
    )

class CRMSchema(BaseModel):  # type: ignore
    prior: List[Probability] = Field(description="Prior probabilities of toxicity")  # type: ignore
    target: Probability = Field(description="Target toxicity probability")  # type: ignore
    first_dose: PositiveInt = Field(default=1, description="First dose level")  # type: ignore
    max_size: PositiveInt = Field(default=30, description="Maximum sample size")  # type: ignore
    lowest_dose_too_toxic_hurdle: Probability = Field(  # type: ignore
        default=0.0, description="Lowest dose hurdle"
    )
    lowest_dose_too_toxic_certainty: Probability = Field(  # type: ignore
        default=0.0, description="Lowest dose certainty"
    )
    coherency_threshold: Probability = Field(  # type: ignore
        default=0.0, description="Coherency threshold"
    )
    bootstrap_samples: PositiveInt = Field(default=200, description="Bootstrap samples")  # type: ignore
    min_beta: Optional[float] = Field(default=None, description="Minimum beta limit")  # type: ignore
    max_beta: Optional[float] = Field(default=None, description="Maximum beta limit")  # type: ignore
    n_points: Optional[PositiveInt] = Field(default=None, description="Integration point count")  # type: ignore
    sample_size: Optional[PositiveInt] = Field(default=None, description="Monte Carlo sample size")  # type: ignore

    def __post_init__(self):  # type: ignore
        super().__post_init__()  # type: ignore
        if self.min_beta is not None and self.max_beta is not None:
            if self.min_beta >= self.max_beta:
                raise ValueError("min_beta must be less than max_beta")

class EffToxSchema(BaseModel):  # type: ignore
    real_doses: List[float] = Field(description="Real dose values")  # type: ignore
    prior_tox_probs: Optional[List[Probability]] = Field(  # type: ignore
        default=None, description="Prior tox probs"
    )
    prior_eff_probs: Optional[List[Probability]] = Field(  # type: ignore
        default=None, description="Prior eff probs"
    )
    tox_cutoff: Optional[Probability] = Field(  # type: ignore
        default=None, description="Toxicity cutoff"
    )
    eff_cutoff: Optional[Probability] = Field(  # type: ignore
        default=None, description="Efficacy cutoff"
    )
    tox_certainty: Optional[Probability] = Field(  # type: ignore
        default=None, description="Toxicity certainty"
    )
    eff_certainty: Optional[Probability] = Field(  # type: ignore
        default=None, description="Efficacy certainty"
    )
    max_size: Optional[PositiveInt] = Field(default=None, description="Maximum size")  # type: ignore
    first_dose: PositiveInt = Field(default=1, description="First dose level")  # type: ignore

