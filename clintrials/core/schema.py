import dataclasses
from typing import Annotated, List, Optional, Union, get_origin, get_args

from clintrials.core.errors import ErrorTemplates

class FieldInfo:
    def __init__(self, default=dataclasses.MISSING, description=None, ge=None, le=None, gt=None, lt=None):
        self.default = default
        self.description = description
        self.ge = ge
        self.le = le
        self.gt = gt
        self.lt = lt
        self.annotation = None

def Field(default=dataclasses.MISSING, description=None, ge=None, le=None, gt=None, lt=None, **kwargs):
    """Create and return a FieldInfo instance."""
    return FieldInfo(default=default, description=description, ge=ge, le=le, gt=gt, lt=lt)

class BaseModel:
    def __init_subclass__(cls, **kwargs):
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
        
    def __post_init__(self):
        for name, f in self.model_fields.items():
            val = getattr(self, name)
            self._validate_value(name, val, f)
            
    def _validate_value(self, name, value, f):
        if value is None:
            return
            
        def check_bounds(v, constraints):
            if constraints.ge is not None and v < constraints.ge:
                raise ValueError(ErrorTemplates.GE.format(name=name, bound=constraints.ge))
            if constraints.le is not None and v > constraints.le:
                raise ValueError(ErrorTemplates.LE.format(name=name, bound=constraints.le))
            if constraints.gt is not None and v <= constraints.gt:
                raise ValueError(ErrorTemplates.GT.format(name=name, bound=constraints.gt))
            if constraints.lt is not None and v >= constraints.lt:
                raise ValueError(ErrorTemplates.LT.format(name=name, bound=constraints.lt))

        if isinstance(value, list):
            for item in value:
                check_bounds(item, f)
                self._validate_annotated(name, item, f.annotation)
        else:
            check_bounds(value, f)
            self._validate_annotated(name, value, f.annotation)

    def _validate_annotated(self, name, value, annotation):
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
                    if is_prob and (value < 0.0 or value > 1.0):
                        raise ValueError(ErrorTemplates.PROBABILITY.format(name=name))
                    elif is_pos_int and value <= 0:
                        raise ValueError(ErrorTemplates.POSITIVE_INTEGER.format(name=name))
                    else:
                        if arg.ge is not None and value < arg.ge:
                            raise ValueError(ErrorTemplates.GE.format(name=name, bound=arg.ge))
                        if arg.le is not None and value > arg.le:
                            raise ValueError(ErrorTemplates.LE.format(name=name, bound=arg.le))
                        if arg.gt is not None and value <= arg.gt:
                            raise ValueError(ErrorTemplates.GT.format(name=name, bound=arg.gt))
                        if arg.lt is not None and value >= arg.lt:
                            raise ValueError(ErrorTemplates.LT.format(name=name, bound=arg.lt))
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

    def __post_init__(self):
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

