# SPDX-License-Identifier: MIT

"""Schema definitions for validating trial design parameters."""

from __future__ import annotations

import dataclasses
from typing import Annotated, Any, List, Optional, Union, get_args, get_origin

from clintrials.validation import (
    validate_bounds,
    validate_positive_integer,
    validate_probability,
)
from clintrials.validation import (
    validate_version as _validate_version,
)


class FieldInfo:
    """Information about a model field's constraints and metadata."""

    def __init__(
        self,
        default: Any = dataclasses.MISSING,
        description: Optional[str] = None,
        ge: Optional[Union[int, float]] = None,
        le: Optional[Union[int, float]] = None,
        gt: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        ui_default: Any = None,
    ) -> None:
        """Initializes field metadata with optional defaults and boundaries."""
        self.default = default
        self.description = description
        self.ge = ge
        self.le = le
        self.gt = gt
        self.lt = lt
        self.ui_default = ui_default
        self.annotation = None


def Field(
    default: Any = dataclasses.MISSING,
    description: Optional[str] = None,
    ge: Optional[Union[int, float]] = None,
    le: Optional[Union[int, float]] = None,
    gt: Optional[Union[int, float]] = None,
    lt: Optional[Union[int, float]] = None,
    ui_default: Any = None,
    **kwargs: Any,
) -> Any:
    """Create and return a FieldInfo instance."""
    return FieldInfo(
        default=default,
        description=description,
        ge=ge,
        le=le,
        gt=gt,
        lt=lt,
        ui_default=ui_default,
    )


class BaseModel:
    """Base schema class with automatic validation of type constraints."""

    model_fields: dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Sets up subclasses with dataclass logic and validates schemas."""
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
                    elif isinstance(val.default, (list, dict, set)):
                        import copy
                        from typing import Callable

                        def make_factory(d: Any) -> Callable[[], Any]:
                            return lambda: copy.deepcopy(d)

                        setattr(
                            cls,
                            name,
                            dataclasses.field(
                                default_factory=make_factory(val.default)
                            ),
                        )
                    else:
                        setattr(cls, name, dataclasses.field(default=val.default))
            else:
                f = FieldInfo()
                f.annotation = ann
                cls.model_fields[name] = f

        dataclasses.dataclass(cls)

    def __post_init__(self) -> None:
        """Validates fields after dataclass initialization."""
        for name, f in self.model_fields.items():
            val = getattr(self, name)
            self._validate_value(name, val, f)

    def _validate_value(self, name: str, value: Any, f: Any) -> None:
        if value is None:
            return

        def check_bounds(v: Any, constraints: Any) -> None:
            if constraints.ge is not None or constraints.le is not None:
                validate_bounds(
                    v,
                    lower=constraints.ge,
                    upper=constraints.le,
                    name=name,
                    exclusive=False,
                )
            if constraints.gt is not None or constraints.lt is not None:
                validate_bounds(
                    v,
                    lower=constraints.gt,
                    upper=constraints.lt,
                    name=name,
                    exclusive=True,
                )

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
                raise ValueError(
                    f"Field '{name}' must be an iterable list rather than a scalar."
                )

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
            is_version = False
            for arg in args[1:]:
                if arg == "Probability":
                    is_prob = True
                elif arg == "PositiveInt":
                    is_pos_int = True
                elif arg == "Version":
                    is_version = True

                if isinstance(arg, FieldInfo):
                    if is_prob:
                        validate_probability(value, name)
                    if is_pos_int:
                        validate_positive_integer(value, name)
                    if is_version:
                        _validate_version(value, name)

                    if arg.ge is not None or arg.le is not None:
                        validate_bounds(
                            value,
                            lower=arg.ge,
                            upper=arg.le,
                            name=name,
                            exclusive=False,
                        )
                    if arg.gt is not None or arg.lt is not None:
                        validate_bounds(
                            value, lower=arg.gt, upper=arg.lt, name=name, exclusive=True
                        )
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
    float,
    "Probability",
    Field(ge=0.0, le=1.0, description="A valid probability between 0 and 1."),
]
PositiveInt = Annotated[
    int, "PositiveInt", Field(gt=0, description="A positive integer.")
]
Version = Annotated[
    str, "Version", Field(description="A PEP 440 compliant version string.")
]


class WinRatioSchema(BaseModel):
    """Schema for validating Win Ratio clinical trial design parameters."""

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
    """Schema for validating Continual Reassessment Method design parameters."""

    prior: List[Probability] = Field(
        ui_default=[0.01, 0.08, 0.15, 0.22, 0.29, 0.36],
        description="Prior probabilities of toxicity",
    )
    target: Probability = Field(
        ui_default=0.30, description="Target toxicity probability"
    )
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
    n_points: Optional[PositiveInt] = Field(
        default=None, description="Integration point count"
    )
    sample_size: Optional[PositiveInt] = Field(
        default=None, description="Monte Carlo sample size"
    )

    def __post_init__(self) -> None:
        """Performs additional validation on CRM parameters."""
        super().__post_init__()
        from clintrials.core.errors import ErrorTemplates

        if self.min_beta is not None and self.max_beta is not None:
            if self.min_beta >= self.max_beta:
                raise ValueError(
                    ErrorTemplates.LT.format(name="min_beta", bound="max_beta")
                )


class EffToxSchema(BaseModel):
    """Schema for validating EffTox design parameters."""

    real_doses: List[float] = Field(
        ui_default=[1.0, 2.0, 3.0, 4.0, 5.0], description="Real dose values"
    )
    theta_priors: Optional[Any] = Field(
        default=None, description="Model parameter priors"
    )
    prior_tox_probs: Optional[List[Probability]] = Field(
        default=None,
        ui_default=[0.05, 0.1, 0.2, 0.3, 0.4],
        description="Prior tox probs",
    )
    prior_eff_probs: Optional[List[Probability]] = Field(
        default=None,
        ui_default=[0.2, 0.4, 0.6, 0.7, 0.8],
        description="Prior eff probs",
    )
    tox_cutoff: Optional[Probability] = Field(
        default=None, ui_default=0.4, description="Toxicity cutoff"
    )
    eff_cutoff: Optional[Probability] = Field(
        default=None, ui_default=0.2, description="Efficacy cutoff"
    )
    tox_certainty: Optional[Probability] = Field(
        default=None, ui_default=0.8, description="Toxicity certainty"
    )
    eff_certainty: Optional[Probability] = Field(
        default=None, ui_default=0.8, description="Efficacy certainty"
    )
    max_size: Optional[PositiveInt] = Field(
        default=None, ui_default=30, description="Maximum size"
    )
    first_dose: PositiveInt = Field(default=1, description="First dose level")

    def __post_init__(self) -> None:
        """Performs additional validation on EffTox parameters and prior distributions."""
        super().__post_init__()
        from clintrials.validation import validate_expected_length

        if self.real_doses is not None:
            expected_len = len(self.real_doses)
            if self.prior_tox_probs is not None:
                validate_expected_length(
                    self.prior_tox_probs, expected_len, "prior_tox_probs"
                )
            if self.prior_eff_probs is not None:
                validate_expected_length(
                    self.prior_eff_probs, expected_len, "prior_eff_probs"
                )

        priors = self.theta_priors
        if (
            priors is None
            and self.real_doses is not None
            and self.prior_tox_probs is not None
            and self.prior_eff_probs is not None
        ):
            from clintrials.dosefinding.efftox import efftox_priors_from_skeleton

            priors = efftox_priors_from_skeleton(
                self.real_doses, self.prior_tox_probs, self.prior_eff_probs
            )

        if priors is not None:
            validate_expected_length(priors, 6, "priors")
            beta_T = priors[1].mean()
            # Check if toxicity is non-decreasing
            if beta_T < 0:
                raise ValueError(
                    "Toxicity prior slope (beta_T) should be non-negative."
                )


class WagesTaitSchema(BaseModel):
    """Schema for validating Wages & Tait design parameters."""

    skeletons: List[List[float]] = Field(
        default=[
            [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
            [0.50, 0.60, 0.50, 0.40, 0.30, 0.20],
            [0.40, 0.50, 0.60, 0.50, 0.40, 0.30],
        ],
        description="A list of efficacy skeletons.",
    )
    prior_tox_probs: List[Probability] = Field(
        default=[0.01, 0.08, 0.15, 0.22, 0.29, 0.36],
        description="A list of prior toxicity probabilities.",
    )
    tox_target: Probability = Field(
        default=0.30, description="The target toxicity rate."
    )
    tox_limit: Probability = Field(
        default=0.33, description="The maximum acceptable toxicity probability."
    )
    eff_limit: Probability = Field(
        default=0.05, description="The minimum acceptable efficacy probability."
    )
    max_size: PositiveInt = Field(
        default=64, description="The maximum number of patients in the trial."
    )
    randomisation_stage_size: PositiveInt = Field(
        default=16,
        description="The number of patients to randomize in the first stage.",
    )
    first_dose: Optional[PositiveInt] = Field(
        default=1, description="The starting dose level (1-based)."
    )

    def __post_init__(self) -> None:
        """Performs additional validation on WagesTait parameters."""
        super().__post_init__()
        from clintrials.core.errors import ErrorTemplates
        from clintrials.validation import validate_expected_length, validate_probability

        if self.tox_target > self.tox_limit:
            raise ValueError(
                ErrorTemplates.LE.format(name="tox_target", bound="tox_limit")
            )
        if len(self.skeletons) == 0:
            raise ValueError("skeletons cannot be empty.")
        expected_len = len(self.prior_tox_probs)
        for idx, skeleton in enumerate(self.skeletons):
            validate_expected_length(skeleton, expected_len, f"skeletons[{idx}]")
            for val in skeleton:
                validate_probability(val, "skeletons")


class WATUSchema(BaseModel):
    """Schema for validating WATU design parameters."""

    skeletons: List[List[float]] = Field(
        default=[
            [0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
            [0.50, 0.60, 0.50, 0.40, 0.30, 0.20],
            [0.40, 0.50, 0.60, 0.50, 0.40, 0.30],
        ],
        description="A list of efficacy skeletons.",
    )
    prior_tox_probs: List[Probability] = Field(
        default=[0.01, 0.08, 0.15, 0.22, 0.29, 0.36],
        description="A list of prior toxicity probabilities.",
    )
    tox_target: Probability = Field(
        default=0.30, description="The target toxicity rate."
    )
    tox_limit: Probability = Field(
        default=0.33, description="The maximum acceptable toxicity probability."
    )
    eff_limit: Probability = Field(
        default=0.05, description="The minimum acceptable efficacy probability."
    )
    max_size: PositiveInt = Field(
        default=64, description="The maximum number of patients in the trial."
    )
    stage_one_size: int = Field(
        default=16, description="The size of the first stage of the trial."
    )
    tox_certainty: Probability = Field(
        default=0.05,
        description="The posterior certainty required that toxicity is less than the cutoff.",
    )
    eff_certainty: Probability = Field(
        default=0.05,
        description="The posterior certainty required that efficacy is greater than the cutoff.",
    )
    first_dose: Optional[PositiveInt] = Field(
        default=1, description="The starting dose level (1-based)."
    )

    def __post_init__(self) -> None:
        """Performs additional validation on WATU parameters."""
        super().__post_init__()
        from clintrials.core.errors import ErrorTemplates
        from clintrials.validation import validate_expected_length, validate_probability

        if self.tox_target > self.tox_limit:
            raise ValueError(
                ErrorTemplates.LE.format(name="tox_target", bound="tox_limit")
            )
        if self.stage_one_size < 0:
            raise ValueError(ErrorTemplates.GE.format(name="stage_one_size", bound=0))
        if len(self.skeletons) == 0:
            raise ValueError("skeletons cannot be empty.")
        expected_len = len(self.prior_tox_probs)
        for idx, skeleton in enumerate(self.skeletons):
            validate_expected_length(skeleton, expected_len, f"skeletons[{idx}]")
            for val in skeleton:
                validate_probability(val, "skeletons")


class GroupSequentialDesignSchema(BaseModel):
    """Schema for validating Group Sequential Design parameters."""

    k: PositiveInt = Field(
        default=3, description="The number of analyses (looks) in the trial."
    )
    alpha: Probability = Field(
        default=0.025, description="The overall one-sided significance level."
    )
    timing: Optional[List[float]] = Field(
        default=None,
        ui_default=[0.33, 0.67, 1.0],
        description="A list of information fractions for each look.",
    )
    n_sims: PositiveInt = Field(
        default=1000, description="The number of simulations to run."
    )
    theta: float = Field(
        default=1.0, description="The treatment effect size parameter."
    )

    def __post_init__(self) -> None:
        """Performs additional validation on GSD parameters."""
        super().__post_init__()
        from clintrials.core.errors import ErrorTemplates

        if self.alpha <= 0.0 or self.alpha >= 1.0:
            raise ValueError(ErrorTemplates.PROBABILITY.format(name="alpha"))
