import warnings

from .efftox.design import (
    EffTox,
    InverseQuadraticCurve,
    LpNormCurve,
    efftox_dtp_detail,
    solve_metrizable_efftox_scenario,
)

warnings.warn(
    "clintrials.dosefinding.efftox module is deprecated; use clintrials.dosefinding.efftox.design",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EffTox",
    "LpNormCurve",
    "InverseQuadraticCurve",
    "efftox_dtp_detail",
    "solve_metrizable_efftox_scenario",
]
