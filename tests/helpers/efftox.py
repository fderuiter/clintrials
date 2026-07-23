from typing import Any, Dict, List

from scipy.stats import norm

from clintrials.dosefinding.efftox import EffTox, LpNormCurve


class EffToxBuilder:
    def __init__(self) -> None:
        self._real_doses = [1.0, 2.0, 4.0, 6.6, 10.0]
        self._tox_cutoff = 0.3
        self._eff_cutoff = 0.5
        self._tox_certainty = 0.1
        self._eff_certainty = 0.1
        self._first_dose = 1
        self._max_size = 39

        # Default Thall 2014 priors
        # These are commonly used in the test suite
        self._theta_priors = [
            norm(loc=-3.173, scale=2.0),
            norm(loc=2.812, scale=2.0),
            norm(loc=1.488, scale=2.0),
            norm(loc=-2.484, scale=2.0),
            norm(loc=1.258, scale=2.0),
            norm(loc=2.071, scale=2.0),
        ]
        self._metric = LpNormCurve(
            0.5, 0.3, 0.7, 0.25
        )
        self._kwargs: Dict[str, Any] = {}

    def with_real_doses(self, real_doses: List[Any]) -> "EffToxBuilder":
        self._real_doses = real_doses
        return self

    def with_cutoffs(self, tox_cutoff: float, eff_cutoff: float) -> "EffToxBuilder":
        self._tox_cutoff = tox_cutoff
        self._eff_cutoff = eff_cutoff
        return self

    def with_certainties(self, tox_certainty: float, eff_certainty: float) -> "EffToxBuilder":
        self._tox_certainty = tox_certainty
        self._eff_certainty = eff_certainty
        return self

    def with_first_dose(self, first_dose: int) -> "EffToxBuilder":
        self._first_dose = first_dose
        return self

    def with_max_size(self, max_size: int) -> "EffToxBuilder":
        self._max_size = max_size
        return self

    def with_theta_priors(self, theta_priors: List[Any]) -> "EffToxBuilder":
        self._theta_priors = theta_priors
        return self

    def with_metric(self, metric: Any) -> "EffToxBuilder":
        self._metric = metric
        return self

    def with_kwargs(self, **kwargs: Any) -> "EffToxBuilder":
        self._kwargs.update(kwargs)
        return self

    def build(self) -> EffTox:
        return EffTox(  # type: ignore[abstract]
            real_doses=self._real_doses,
            tox_cutoff=self._tox_cutoff,
            eff_cutoff=self._eff_cutoff,
            tox_certainty=self._tox_certainty,
            eff_certainty=self._eff_certainty,
            first_dose=self._first_dose,
            max_size=self._max_size,
            theta_priors=self._theta_priors,
            metric=self._metric,
            **self._kwargs
        )
