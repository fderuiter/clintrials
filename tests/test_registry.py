# SPDX-License-Identifier: MIT

from typing import Any, Dict

import pytest

from clintrials.core.protocol import Protocol
from clintrials.core.registry import RUNNER_REGISTRY


class LookupTestTrial(Protocol):
    """Simple protocol design subclass for testing runner registries."""

    def __init__(self) -> None:
        super().__init__()
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def has_more(self) -> bool:
        return False

    def report(self) -> Dict[str, str]:
        return {"status": "ok"}


class CustomTestRunner:
    """Custom runner for registry testing with custom configuration mapping."""

    config_mapping = {"multiplier": "times"}

    def __init__(self, design: Any) -> None:
        self.design = design
        self.rng = None

    def set_rng(self, rng: Any) -> None:
        self.rng = rng

    def run(
        self, mode: str, n_sims: int, show_progress: bool, times: int = 1, **kwargs: Any
    ) -> Dict[str, Any]:
        return {
            "mode": mode,
            "n_sims": n_sims,
            "times": times,
            "rng_set": self.rng is not None,
        }


class CustomTestResult:
    """Custom results container wrapping runner results."""

    def __init__(self, results: Any, mode: str) -> None:
        self.results = results
        self.mode = mode


def test_custom_runner_lookup_and_execution() -> None:
    """Verify that a custom registered runner successfully overrides defaults and executes correctly."""
    # Register custom runner and result container
    RUNNER_REGISTRY.register(LookupTestTrial, CustomTestRunner, CustomTestResult)

    try:
        trial = LookupTestTrial()
        # Execute with custom keyword argument "multiplier" which is mapped to "times" by config_mapping
        result = trial.run(n_sims=5, method="bulk", seed=123, multiplier=3)

        # Verify custom result container was returned
        assert isinstance(result, CustomTestResult)
        assert result.mode == "bulk"

        # Verify custom runner executed and received mapped kwargs and injected RNG
        assert result.results["mode"] == "vectorized"
        assert result.results["n_sims"] == 5
        assert result.results["times"] == 3
        assert result.results["rng_set"] is True
    finally:
        # Clean up registration
        RUNNER_REGISTRY._registry.pop(LookupTestTrial, None)


def test_custom_runner_decorator_registration() -> None:
    """Verify registry decorator-based registration style works correctly."""

    class DecoratorTestTrial(Protocol):
        def reset(self) -> None:
            pass

        def update(self, *args: Any, **kwargs: Any) -> None:
            pass

        def has_more(self) -> bool:
            return False

        def report(self) -> Dict[str, str]:
            return {}

    @RUNNER_REGISTRY.register(DecoratorTestTrial)
    class DecoratorRunner:
        def __init__(self, design: Any) -> None:
            self.design = design

        def run(
            self, mode: str, n_sims: int, show_progress: bool, **kwargs: Any
        ) -> str:
            return "decorated_success"

    try:
        trial = DecoratorTestTrial()
        result = trial.run(n_sims=1)
        assert result.results == "decorated_success"
    finally:
        RUNNER_REGISTRY._registry.pop(DecoratorTestTrial, None)


def test_invalid_registration_validation() -> None:
    """Verify that invalid registration calls throw explicit validation errors with clear troubleshooting instructions."""

    # 1. Design is None
    with pytest.raises(ValueError) as exc_info_val:
        RUNNER_REGISTRY.register(None, CustomTestRunner)
    assert "Unable to register: The 'design' parameter cannot be None." in str(
        exc_info_val.value
    )
    assert "Troubleshooting:" in str(exc_info_val.value)

    # 2. Design is wrong type (e.g., list or integer)
    with pytest.raises(TypeError) as exc_info_type:
        RUNNER_REGISTRY.register(123, CustomTestRunner)
    assert "Unable to register: Invalid 'design' parameter type 'int'." in str(
        exc_info_type.value
    )
    assert "Troubleshooting:" in str(exc_info_type.value)

    # 3. Runner is wrong type (non-callable)
    with pytest.raises(TypeError) as exc_info_runner:
        RUNNER_REGISTRY.register(LookupTestTrial, runner="not_callable")
    assert "Unable to register: The 'runner' parameter must be callable" in str(
        exc_info_runner.value
    )
    assert "Troubleshooting:" in str(exc_info_runner.value)

    # 4. Result container is not callable
    with pytest.raises(TypeError) as exc_info_res:
        RUNNER_REGISTRY.register(
            LookupTestTrial, runner=CustomTestRunner, result_container="not_callable"
        )
    assert (
        "Unable to register: The 'result_container' parameter must be callable"
        in str(exc_info_res.value)
    )
    assert "Troubleshooting:" in str(exc_info_res.value)


def test_config_map_callable_support() -> None:
    """Verify custom config_map callable works correctly."""

    class CallableMapTrial(Protocol):
        def reset(self) -> None:
            pass

        def update(self, *args: Any, **kwargs: Any) -> None:
            pass

        def has_more(self) -> bool:
            return False

        def report(self) -> Dict[str, str]:
            return {}

    class CustomCallableRunner:
        # custom config_map as callable
        @staticmethod
        def config_map(kwargs: Dict[str, Any]) -> Dict[str, Any]:
            new_kwargs = kwargs.copy()
            if "factor" in new_kwargs:
                new_kwargs["scale"] = new_kwargs.pop("factor") * 10
            return new_kwargs

        def __init__(self, design: Any) -> None:
            self.design = design

        def run(
            self,
            mode: str,
            n_sims: int,
            show_progress: bool,
            scale: int = 0,
            **kwargs: Any,
        ) -> Dict[str, int]:
            return {"scale": scale}

    RUNNER_REGISTRY.register(CallableMapTrial, CustomCallableRunner)
    try:
        trial = CallableMapTrial()
        result = trial.run(n_sims=1, factor=5)
        assert result.results["scale"] == 50
    finally:
        RUNNER_REGISTRY._registry.pop(CallableMapTrial, None)
