import importlib
import logging

from collections import defaultdict
from packaging.version import Version

from compehndly.core.conversion import arrowize_arguments
from compehndly.adapters import _ADAPTERS

logger = logging.getLogger(__name__)

TO_REGISTER = [
    "compehndly.derived_variables.correction",
    "compehndly.derived_variables.imputation",
    "compehndly.derived_variables.summation",
]


class FunctionRegistry:
    def __init__(self, adapter: str | None = None):
        self._functions = defaultdict(dict)
        logging.debug("Running function registry")
        if adapter is None:
            self.adapter = _ADAPTERS["base"]
        else:
            if adapter not in _ADAPTERS:
                raise ValueError(f"Unknown adapter '{adapter}'. " f"Available: {', '.join(_ADAPTERS)}")
            self.adapter = _ADAPTERS[adapter]

    def register(self, name, version, func):
        version = Version(version)
        if version in self._functions[name]:
            raise ValueError(f"Function {name} version {version} already registered.")

        wrapped_func = arrowize_arguments(func, adapter=self.adapter)
        self._functions[name][version] = wrapped_func

    def get(self, name, version=None):
        """Return the function. If version is None, return latest."""
        if name not in self._functions:
            raise KeyError(f"No function registered with name '{name}'")
        versions = sorted(self._functions[name].keys())
        if version is None:
            version = versions[-1]
        else:
            version = Version(version)
        return self._functions[name][version]

    def list_versions(self, name):
        if name not in self._functions:
            return []
        return sorted(str(v) for v in self._functions[name].keys())

    @classmethod
    def build_registry(cls, _to_register=TO_REGISTER, adapter: str | None = None):
        registry = cls(adapter=adapter)

        for module_path in _to_register:
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise ImportError(f"Failed to import module '{module_path}': {e}")

            if not hasattr(module, "__registrations__") or not isinstance(module.__registrations__, list):
                continue

            for registry_name, func_name, version, func in module.__registrations__:
                if registry_name != "default":
                    raise ValueError(f"Unsupported registry name '{registry_name}' in module '{module_path}'")
                registry.register(func_name, version, func)

        return registry
