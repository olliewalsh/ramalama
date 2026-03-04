from dataclasses import MISSING, fields, is_dataclass
from functools import reduce
from typing import Any, Type, get_type_hints

from ramalama.logger import logger


def deep_merge(left: dict, right: dict) -> dict:
    """Recursively merge override dict into base dict."""
    for key, value in right.items():
        if isinstance(left_value := left.get(key, None), dict) and isinstance(value, dict):
            left[key] = deep_merge(left_value, value)
        else:
            left[key] = value
    return left


def extract_defaults(cls) -> dict[str, Any]:
    result = {}
    for f in fields(cls):
        if f.default is not MISSING:
            result[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore
            result[f.name] = f.default_factory()  # type: ignore
    return result


def build_subconfigs(values: dict, obj: Type) -> dict:
    """Facilitates nesting configs by instantiating the child typed object from a dict

    NOTE: This implementation does not automatically coerce more complicated config structures
    involving types like (ConfigObj | None), or list[ConfigObj], etc...
    """

    dtypes: dict[str, Type] = get_type_hints(obj)
    for k, v in values.items():
        if isinstance(v, dict) and (subconfig_type := dtypes.get(k)) and is_dataclass(subconfig_type):
            values[k] = subconfig_type(**build_subconfigs(v, dtypes[k]))

    return values


class LayeredMixin:
    """Mixin class to handle layered configurations from multiple sources."""

    _layers_finalized = False

    def __init__(self, *layers: dict[str, Any]):
        self._fields = {f.name for f in fields(self.__class__) if not f.name.startswith("_")}  # type: ignore[arg-type]
        self._layers = [{k: layer[k] for k in layer.keys() & self._fields} for layer in layers]

        merged = extract_defaults(self.__class__)
        if self._layers:
            merged |= reduce(deep_merge, self._layers)  # type: ignore[arg-type]
        super().__init__(**build_subconfigs(merged, type(self)))
        # Add an empty layer to store values set via the instance attributes
        self._layers.append({})
        self._layers_finalized = True

    def is_set(self, name: str) -> bool:
        """Returns True if the config attribute is explicitly set vs the default value."""
        return object.__getattribute__(self, "_layers_finalized") and any(
            name in layer for layer in object.__getattribute__(self, "_layers")
        )

    def __setattr__(self, name: str, value: Any):
        if object.__getattribute__(self, "_layers_finalized"):
            if name not in self._fields:
                raise AttributeError(f"Attribute {name} not found in config class {self.__class__.__name__}")
            logger.debug(f"Setting config attribute {name} to {value}")
            object.__getattribute__(self, "_layers")[-1][name] = value
        super().__setattr__(name, value)
