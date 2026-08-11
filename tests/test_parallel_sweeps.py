# SPDX-License-Identifier: MIT

from clintrials.core.simulation import run_sims, sim_parameter_space
from clintrials.utils import ParameterSpace


def stochastic_sim(seed=None):
    from clintrials.core.rng import get_rng
    rng = get_rng(seed)
    val = rng.uniform(0, 100)
    ints = rng.integers(0, 1000)
    return {"val": val, "ints": ints, "seed_used": seed}


def param_stochastic_sim(param1, seed=None):
    from clintrials.core.rng import get_rng
    rng = get_rng(seed)
    val = rng.uniform(0, 100) * param1
    return {"val": val, "param1": param1, "seed_used": seed}


def test_run_sims_parallel_reproducibility():
    """Verify that parallel and sequential runs of run_sims with seed produce identical results."""
    res_seq = run_sims(stochastic_sim, n1=2, n2=3, seed=42, parallel=False)
    res_par = run_sims(stochastic_sim, n1=2, n2=3, seed=42, parallel=True, n_workers=2)

    assert len(res_seq) == 6
    assert len(res_par) == 6
    for s_item, p_item in zip(res_seq, res_par):
        assert s_item["val"] == p_item["val"]
        assert s_item["ints"] == p_item["ints"]
        assert s_item["seed_used"] == p_item["seed_used"]


def test_sim_parameter_space_parallel_reproducibility():
    """Verify that parallel and sequential runs of sim_parameter_space produce identical results."""
    ps = ParameterSpace()
    ps.add("param1", [1.5, 2.5, 3.5])
    ps.add("seed", [100])

    res_seq = sim_parameter_space(param_stochastic_sim, ps, n1=2, n2=3, parallel=False)
    res_par = sim_parameter_space(param_stochastic_sim, ps, n1=2, n2=3, parallel=True, n_workers=2)

    assert len(res_seq) == 6
    assert len(res_par) == 6
    for s_item, p_item in zip(res_seq, res_par):
        assert s_item["val"] == p_item["val"]
        assert s_item["param1"] == p_item["param1"]
        assert s_item["seed_used"] == p_item["seed_used"]


def test_parallel_toggle_and_workers():
    """Verify that specifying n_workers or parallel flag behaves as expected without errors."""
    # parallel=True, n_workers=1 should fall back to sequential execution
    res1 = run_sims(stochastic_sim, n1=1, n2=2, seed=123, parallel=True, n_workers=1)
    res_seq = run_sims(stochastic_sim, n1=1, n2=2, seed=123, parallel=False)
    assert res1 == res_seq

    # parallel=False, n_workers=2 should run sequentially
    res2 = run_sims(stochastic_sim, n1=1, n2=2, seed=123, parallel=False, n_workers=2)
    assert res2 == res_seq

    # parallel=True with default workers should run successfully in parallel
    res3 = run_sims(stochastic_sim, n1=1, n2=2, seed=123, parallel=True)
    assert res3 == res_seq
