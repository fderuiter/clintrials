# SPDX-License-Identifier: MIT

import inspect
import os
import tempfile
from typing import Any, Dict, List, Optional

from clintrials.core.simulation import run_sims
from clintrials.utils import Memoize
from clintrials.visualization.dashboard.main import (
    get_design_file_mtimes,
)


def test_sequence_types_distinguished() -> None:
    """Verify sequence types (lists vs tuples) are distinguished and do not collide in Memoize."""
    call_count = 0

    @Memoize
    def func(seq: Any) -> int:
        nonlocal call_count
        call_count += 1
        return len(seq)

    # Calling with list
    assert func([1, 2]) == 2
    assert call_count == 1

    # Calling with list again (hit cache)
    assert func([1, 2]) == 2
    assert call_count == 1

    # Calling with tuple (should be a cache miss because type is different)
    assert func((1, 2)) == 2
    assert call_count == 2


def test_recycled_memory_address_no_collision(mocker: Any) -> None:
    """Verify that custom objects with cycles or falling back to str don't collide if recycled."""

    class CustomObj:
        def __init__(self, val: int) -> None:
            self.val = val
            self.ref = self  # Cycle!

    obj1 = CustomObj(10)
    obj2 = CustomObj(20)

    # Mock id() to simulate recycled memory address
    orig_id = id

    def mock_id(obj: Any) -> int:
        if isinstance(obj, CustomObj):
            return 888888
        return orig_id(obj)

    mocker.patch("clintrials.utils.id", side_effect=mock_id)

    call_count = 0

    @Memoize
    def process(obj: Any) -> int:
        nonlocal call_count
        call_count += 1
        return int(obj.val)

    assert process(obj1) == 10
    assert call_count == 1

    # obj2 has same simulated ID, but different attributes.
    # Because unique ID counter is stable, they shouldn't collide.
    assert process(obj2) == 20
    assert call_count == 2


def test_seed_bypass() -> None:
    """Verify running simulation with seed caches, but without seed bypasses."""

    class Tracker:
        call_count = 0

    def dummy_sim(seed: Optional[int] = None) -> Dict[str, int]:
        Tracker.call_count += 1
        return {"result": Tracker.call_count}

    # run_sims is decorated with @Memoize
    # 1. Run without seed: should bypass cache
    res1 = run_sims(dummy_sim, n1=1, n2=1)
    assert res1 == [{"result": 1}]
    assert Tracker.call_count == 1

    # Run without seed again: should bypass cache and return fresh result
    res2 = run_sims(dummy_sim, n1=1, n2=1)
    assert res2 == [{"result": 2}]
    assert Tracker.call_count == 2

    # 2. Run with seed: should cache
    res3 = run_sims(dummy_sim, n1=1, n2=1, seed=42)
    assert res3 == [{"result": 3}]
    assert Tracker.call_count == 3

    # Run with same seed again: should hit cache and return cached result
    res4 = run_sims(dummy_sim, n1=1, n2=1, seed=42)
    assert res4 == [{"result": 3}]
    assert Tracker.call_count == 3  # call_count remains 3


def test_maximum_cache_size_cap() -> None:
    """Verify Memoize strictly caps max size below 128 (e.g. at 32)."""

    @Memoize(maxsize=128)
    def my_func(x: int) -> int:
        return x

    assert my_func.maxsize <= 32


def test_dashboard_preview_cache_file_mtimes(monkeypatch: Any) -> None:
    """Verify that updating a design file modification time triggers a fresh preview calculation."""
    # Create temp view and design files to simulate change
    with tempfile.TemporaryDirectory() as tmpdir:
        view_path = os.path.join(tmpdir, "mock_view.py")
        model_path = os.path.join(tmpdir, "mock_model.py")

        with open(view_path, "w") as f:
            f.write("class MockView: pass")
        with open(model_path, "w") as f:
            f.write("class MockModel: pass")

        # Set up a mock preview function that behaves like bound view class method
        class MockViewClass:
            model_class: Any = None

            @classmethod
            def preview(
                cls, target_tox: float, cohort_size: int, max_size: int
            ) -> List[Dict[str, int]]:
                return [{"data": 123}]

        MockViewClass.model_class = type("MockModelClass", (), {})

        # Patch inspect.getfile to return our temp paths
        orig_getfile = inspect.getfile

        def mock_getfile(obj: Any) -> str:
            if obj is MockViewClass.preview:
                return view_path
            if obj is MockViewClass.model_class:
                return model_path
            return orig_getfile(obj)

        monkeypatch.setattr(inspect, "getfile", mock_getfile)

        # Mock PROTOCOL_REGISTRY.get_preview to return our mock preview
        from clintrials.core.registry import PROTOCOL_REGISTRY

        monkeypatch.setattr(
            PROTOCOL_REGISTRY, "get_preview", lambda design: MockViewClass.preview
        )

        # 1. Get initial mtimes
        mtimes1 = get_design_file_mtimes("Mock")

        # 2. Wait or update mtimes manually to simulate a file edit
        os.utime(
            model_path,
            (os.path.getatime(model_path), os.path.getmtime(model_path) + 10.0),
        )
        mtimes2 = get_design_file_mtimes("Mock")

        assert mtimes1["model"] != mtimes2["model"]
