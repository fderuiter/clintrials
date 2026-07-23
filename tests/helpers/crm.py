import numpy as np
from scipy.stats import norm

from clintrials.core.math import logistic, inverse_logistic
from clintrials.dosefinding.crm import CRM


class CRMBuilder:
    def __init__(self) -> None:
        self._prior = [0.05, 0.12, 0.25, 0.40, 0.55]
        self._target = 0.25
        self._first_dose = 3
        self._max_size = 30
        self._F_func = logistic
        self._inverse_F = inverse_logistic
        self._beta_prior = norm(loc=0, scale=np.sqrt(1.34))
        self._method = "bayes"
        self._use_quick_integration = False
        self._estimate_var = True
        self._mle_var_method = "hessian"
        self._kwargs = {}

    def with_prior(self, prior: list) -> "CRMBuilder":
        self._prior = prior
        return self

    def with_target(self, target: float) -> "CRMBuilder":
        self._target = target
        return self

    def with_first_dose(self, first_dose: int) -> "CRMBuilder":
        self._first_dose = first_dose
        return self

    def with_max_size(self, max_size: int) -> "CRMBuilder":
        self._max_size = max_size
        return self

    def with_F_func(self, F_func, inverse_F=None) -> "CRMBuilder":
        self._F_func = F_func
        self._inverse_F = inverse_F
        return self

    def with_beta_prior(self, beta_prior) -> "CRMBuilder":
        self._beta_prior = beta_prior
        return self

    def with_method(self, method: str) -> "CRMBuilder":
        self._method = method
        return self

    def with_use_quick_integration(self, use_quick_integration: bool) -> "CRMBuilder":
        self._use_quick_integration = use_quick_integration
        return self

    def with_estimate_var(self, estimate_var: bool) -> "CRMBuilder":
        self._estimate_var = estimate_var
        return self

    def with_mle_var_method(self, mle_var_method: str) -> "CRMBuilder":
        self._mle_var_method = mle_var_method
        return self

    def with_kwargs(self, **kwargs) -> "CRMBuilder":
        self._kwargs.update(kwargs)
        return self

    def build(self) -> CRM:
        return CRM(
            prior=self._prior,
            target=self._target,
            first_dose=self._first_dose,
            max_size=self._max_size,
            F_func=self._F_func,
            inverse_F=self._inverse_F,
            beta_prior=self._beta_prior,
            method=self._method,
            use_quick_integration=self._use_quick_integration,
            estimate_var=self._estimate_var,
            mle_var_method=self._mle_var_method,
            **self._kwargs
        )
