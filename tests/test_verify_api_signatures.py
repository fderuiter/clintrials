import typing

from scripts.verify_api_signatures import (
    clean_dynamic_info,
    compare_manifests,
    compare_parameters,
    get_public_methods,
    get_signature_info,
)


def test_clean_dynamic_info() -> None:
    assert clean_dynamic_info(None) is None
    assert clean_dynamic_info("abc at 0x7f3a5474f200 xyz") == "abc at 0x... xyz"


def test_get_signature_info() -> None:
    def dummy_func(a: int, b: str = "test", *, c: float = 1.0) -> None:
        pass

    sig_info = get_signature_info(dummy_func)
    assert len(sig_info) == 3

    assert sig_info[0]["name"] == "a"
    assert sig_info[0]["annotation"] == "int"
    assert sig_info[0]["kind"] == "POSITIONAL_OR_KEYWORD"
    assert sig_info[0]["default"] is None

    assert sig_info[1]["name"] == "b"
    assert sig_info[1]["annotation"] == "str"
    assert sig_info[1]["default"] == "'test'"

    assert sig_info[2]["name"] == "c"
    assert sig_info[2]["annotation"] == "float"
    assert sig_info[2]["kind"] == "KEYWORD_ONLY"
    assert sig_info[2]["default"] == "1.0"


def test_get_public_methods() -> None:
    class DummyClass:

        def __init__(self, x: int) -> None:
            self.x = x

        def normal_method(self, y: int) -> None:
            pass

        @classmethod
        def class_m(cls, z: str) -> None:
            pass

        @staticmethod
        def static_m(w: float) -> None:
            pass

        @property
        def prop(self) -> int:
            return self.x

    methods = get_public_methods(DummyClass)

    assert "__init__" in methods
    assert methods["__init__"]["type"] == "method"
    assert len(methods["__init__"]["parameters"]) == 2  # self, x

    assert "normal_method" in methods
    assert methods["normal_method"]["type"] == "method"

    assert "class_m" in methods
    assert methods["class_m"]["type"] == "classmethod"
    assert methods["class_m"]["parameters"][0]["name"] == "cls"

    assert "static_m" in methods
    assert methods["static_m"]["type"] == "staticmethod"
    assert methods["static_m"]["parameters"][0]["name"] == "w"

    assert "prop" in methods
    assert methods["prop"]["type"] == "property"
    assert "parameters" not in methods["prop"]


def test_compare_parameters() -> None:
    # Identical parameters
    p1: typing.List[typing.Dict[str, typing.Any]] = [
        {"name": "a", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "int", "default": None}
    ]
    p2: typing.List[typing.Dict[str, typing.Any]] = [
        {"name": "a", "kind": "POSITIONAL_OR_KEYWORD", "annotation": "int", "default": None}
    ]
    assert compare_parameters(p1, p2) == []

    # Missing/removed parameter
    p3: typing.List[typing.Dict[str, typing.Any]] = []
    diffs = compare_parameters(p1, p3)
    assert any("removed" in d for d in diffs)

    # Added parameter
    diffs2 = compare_parameters(p3, p1)
    assert any("added" in d for d in diffs2)

    # Modified parameter (annotation, default, kind)
    p4: typing.List[typing.Dict[str, typing.Any]] = [
        {"name": "a", "kind": "KEYWORD_ONLY", "annotation": "str", "default": "'hello'"}
    ]
    diffs3 = compare_parameters(p1, p4)
    assert len(diffs3) > 0
    assert any("kind changed" in d for d in diffs3)
    assert any("annotation changed" in d for d in diffs3)
    assert any("default value changed" in d for d in diffs3)

    # Order changed
    p_order1: typing.List[typing.Dict[str, typing.Any]] = [
        {"name": "a", "kind": "POSITIONAL_OR_KEYWORD", "annotation": None, "default": None},
        {"name": "b", "kind": "POSITIONAL_OR_KEYWORD", "annotation": None, "default": None},
    ]
    p_order2: typing.List[typing.Dict[str, typing.Any]] = [
        {"name": "b", "kind": "POSITIONAL_OR_KEYWORD", "annotation": None, "default": None},
        {"name": "a", "kind": "POSITIONAL_OR_KEYWORD", "annotation": None, "default": None},
    ]
    diffs4 = compare_parameters(p_order1, p_order2)
    assert any("order changed" in d for d in diffs4)


def test_compare_manifests() -> None:
    # Test class changed to function type mismatch without crashing
    baseline: typing.Dict[str, typing.Any] = {
        "module1": {
            "WATU": {
                "type": "class",
                "methods": {
                    "__init__": {
                        "type": "method",
                        "parameters": [
                            {"name": "self", "kind": "POSITIONAL_OR_KEYWORD", "annotation": None, "default": None}
                        ],
                    }
                },
            }
        }
    }
    current: typing.Dict[str, typing.Any] = {
        "module1": {
            "WATU": {
                "type": "function",
                "parameters": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "annotation": None, "default": None}],
            }
        }
    }
    diffs = compare_manifests(baseline, current)
    assert any("type changed from class to function" in d for d in diffs)

    # Test missing module
    diffs_missing_mod = compare_manifests(baseline, {})
    assert any("Module 'module1' is missing" in d for d in diffs_missing_mod)

    # Test missing export
    diffs_missing_exp = compare_manifests(baseline, {"module1": {}})
    assert any("Export 'WATU' missing" in d for d in diffs_missing_exp)
