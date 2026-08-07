# SPDX-License-Identifier: MIT


from typing import Any, Optional

from clintrials.utils import Memoize


def test_memoize():

    class MyClass:
        def __init__(self):
            self.call_count = 0

        @Memoize
        def my_method(self, x):
            self.call_count += 1
            return x * 2

    c = MyClass()  # type: ignore
    assert c.my_method(2) == 4
    assert c.call_count == 1
    assert c.my_method(2) == 4
    assert c.call_count == 1
    assert c.my_method(3) == 6
    assert c.call_count == 2


import pytest

from clintrials.utils import deprecated


def test_deprecated_function():
    @deprecated(alternative="new_func")  # type: ignore
    def old_func():
        return 42

    with pytest.warns(DeprecationWarning) as record:
        result = old_func()

    assert result == 42
    assert len(record) == 1
    assert "old_func is deprecated" in str(record[0].message)
    assert "Use new_func instead" in str(record[0].message)


def test_deprecated_class():
    @deprecated(alternative="NewClass")  # type: ignore
    class OldClass:
        def __init__(self, val):
            self.val = val

    with pytest.warns(DeprecationWarning) as record:
        obj = OldClass(10)  # type: ignore

    assert obj.val == 10
    assert len(record) == 1
    assert "OldClass is deprecated" in str(record[0].message)
    assert "Use NewClass instead" in str(record[0].message)


def test_memoize_no_false_cache_hit_on_recycled_id(mocker: Any) -> None:
    class Parameter:
        def __init__(self, value: int) -> None:
            self.value = value

    p1 = Parameter(10)
    p2 = Parameter(20)

    # Mock id() to return the same address for both parameters to simulate recycled memory
    original_id = id

    def mock_id(obj):
        if isinstance(obj, Parameter):
            return 999999
        return original_id(obj)

    mocker.patch("clintrials.utils.id", side_effect=mock_id)

    call_count = 0

    @Memoize
    def my_func(p):
        nonlocal call_count
        call_count += 1
        return p.value * 2

    assert my_func(p1) == 20
    assert call_count == 1

    # In our robust version, since p2's dict is serialized (value: 20),
    # it must result in a different cache key despite having the same id, and trigger another call.
    assert my_func(p2) == 40
    assert call_count == 2


def test_memoize_fallback_on_serialization_failure() -> None:
    class UnserializableParameter:
        @property
        def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
            raise ValueError("Serialization failed!")

    p = UnserializableParameter()

    call_count = 0

    @Memoize
    def my_func(x):
        nonlocal call_count
        call_count += 1
        return 42

    # Should gracefully fallback to direct execution
    assert my_func(p) == 42
    assert call_count == 1

    # Subsequent call also falls back and calls directly
    assert my_func(p) == 42
    assert call_count == 2


def test_memoize_instance_binding_and_leak_prevention() -> None:
    class MyClass:
        def __init__(self, multiplier: int) -> None:
            self.multiplier = multiplier
            self.calls = 0

        @Memoize(maxsize=2)
        def compute(self, x: int) -> int:
            self.calls += 1
            return x * self.multiplier

    obj1 = MyClass(2)
    obj2 = MyClass(3)

    assert obj1.compute(5) == 10
    assert obj1.calls == 1

    # obj2 is separate, must not hit obj1's cache
    assert obj2.compute(5) == 15
    assert obj2.calls == 1

    # obj1 again, must hit cache
    assert obj1.compute(5) == 10
    assert obj1.calls == 1

    # Verify maxsize eviction (maxsize=2)
    assert obj1.compute(6) == 12  # cache has 5, 6
    assert obj1.compute(7) == 14  # cache has 6, 7 (5 is evicted)
    assert obj1.calls == 3

    # Calling 5 again must be a miss
    assert obj1.compute(5) == 10
    assert obj1.calls == 4


def test_memoize_cycle_detection() -> None:
    class CyclicNode:
        def __init__(self, name: str) -> None:
            self.name = name
            self.ref: Optional[CyclicNode] = None

    node1 = CyclicNode("A")
    node2 = CyclicNode("B")
    node1.ref = node2
    node2.ref = node1

    call_count = 0

    @Memoize
    def process_node(node):
        nonlocal call_count
        call_count += 1
        return len(node.name)

    assert process_node(node1) == 1
    assert call_count == 1

    # Second call should safely hit the cache even with cycle
    assert process_node(node1) == 1
    assert call_count == 1
