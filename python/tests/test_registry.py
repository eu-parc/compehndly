import compehndly
import pytest

from typing import Callable


@pytest.mark.core
class TestTestRegistry:
    def test_build(self):
        to_register = ["tests.utils"]
        registry = compehndly.FunctionRegistry.build_registry(to_register)
        f = registry.get("add_one", version="0.0.1")
        x = 10
        assert f(x) == 11


class TestFunctionAccessor:
    def test_simple(self):
        to_register = ["tests.utils"]
        compehndly._set_registry_builder(lambda: compehndly.FunctionRegistry.build_registry(_to_register=to_register))
        f = compehndly.add_one["0.0.1"]
        assert callable(f)


class TestTrueRegistry:
    f = compehndly.medium_bound_imputation["0.0.1"]
    assert isinstance(f, Callable)
