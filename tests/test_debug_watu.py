import numpy as np
from clintrials.dosefinding.efftox import LpNormCurve
from clintrials.dosefinding.watu import WATU

def test_watu_simple_update():
    tox_prior = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    tox_cutoff = 0.4
    eff_cutoff = 0.1
    tox_target = 0.2
    skeletons = [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]
    metric = LpNormCurve(0.1, 0.4, 0.2, 0.2)
    trial = WATU(skeletons, tox_prior, tox_target, tox_cutoff, eff_cutoff, metric,
                 first_dose=1, max_size=30, stage_one_size=10, tox_certainty=0.0001, eff_certainty=0.0001)
    print(f"Initial next dose: {trial.next_dose()}")
    res = trial.update([(1, 0, 1)])
    print(f"Updated next dose: {res}")
    print(f"Admissable set: {trial.admissable_set()}")
    print(f"Prob acc tox: {trial.prob_acc_tox(0.4)}")
    print(f"Prob acc eff: {trial.prob_acc_eff(0.1)}")
    assert res != -1
