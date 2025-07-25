import warnings

from .crm.design import CRM, crm, crm_dtp_detail

warnings.warn(
    "clintrials.dosefinding.crm module is deprecated; use clintrials.dosefinding.crm.design",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CRM", "crm", "crm_dtp_detail"]
