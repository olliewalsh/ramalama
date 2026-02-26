import argparse
from abc import ABC, abstractmethod
from http.client import HTTPConnection
from typing import Any


class RuntimePlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    def get_container_image(self, config: Any, gpu_type: str) -> str | None:
        return None

    def setup_args(self, args: argparse.Namespace) -> None:
        pass

    def validate_args(self, args: Any) -> None:
        """Validate runtime-specific argument constraints. Raise ValueError if invalid."""
        pass

    def register_subcommands(self, subparsers: "argparse._SubParsersAction") -> None:
        """Register runtime-specific subcommand parsers.

        Override to add subcommands that only apply to this runtime.
        Called from configure_subcommands() after universal commands are registered.
        """
        pass

    def handle_subcommand(self, command: str, args: argparse.Namespace) -> list[str]:
        """Handle the given subcommand. Override in concrete plugins."""
        raise NotImplementedError(f"{self.name} plugin does not implement handle_subcommand()")

    @property
    def health_check_timeout(self) -> int:
        """Seconds to wait for the runtime server to become healthy."""
        return 20

    def is_healthy(self, conn: HTTPConnection, args: Any, model_name: str | None = None) -> bool:
        """Check service health. Override to implement runtime-specific health checks."""
        raise NotImplementedError(f"Runtime plugin '{self.name}' does not implement is_healthy()")


class InferenceRuntimePlugin(RuntimePlugin, ABC):
    """Abstract base class for runtime plugins that support 'run' and 'serve' subcommands.

    Concrete subclasses must implement _cmd_run.  The hooks below have safe
    no-op defaults and can be overridden to customise per-runtime behaviour.
    """

    @abstractmethod
    def _cmd_run(self, args: argparse.Namespace) -> list[str]:
        """Build the command list for the 'run' subcommand."""

    def api_model_name(self, args: Any) -> "str | None":
        """Return the model name to include in API requests, or None to omit it."""
        return getattr(args, "model", None)
