from ramalama.plugins.base import RuntimePlugin
from ramalama.plugins.registry import PluginRegistry

# Registry for runtime plugins (entry point group: "ramalama.runtimes")
# Future plugin types register their own PluginRegistry with a distinct group name, e.g.:
#   CHAT_PLUGIN_REGISTRY = PluginRegistry("ramalama.chat_providers", ChatPlugin)
_RUNTIME_REGISTRY: PluginRegistry[RuntimePlugin] = PluginRegistry("ramalama.runtimes", RuntimePlugin)  # type: ignore[type-abstract]


def get_all_runtimes() -> dict[str, RuntimePlugin]:
    return _RUNTIME_REGISTRY.load()


def get_runtime(name: str) -> RuntimePlugin | None:
    return _RUNTIME_REGISTRY.get(name)
