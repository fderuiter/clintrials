from pydantic import BaseModel, Field
from typing import Annotated

Probability = Annotated[
    float, Field(ge=0.0, le=1.0, description="A valid probability between 0 and 1.")
]
PositiveInt = Annotated[int, Field(gt=0, description="A positive integer.")]


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


from typing import List, Optional


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
