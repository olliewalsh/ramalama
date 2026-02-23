import argparse
from abc import ABC, abstractmethod
from http.client import HTTPConnection
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ramalama.command.context import RamalamaCommandContext


class RuntimePlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def build_command(self, command: str, ctx: "RamalamaCommandContext") -> list[str]: ...

    def get_container_image(self, config: Any, gpu_type: str) -> str | None:
        return None

    def setup_args(self, args: argparse.Namespace) -> None:
        pass

    def register_subcommands(self, subparsers: "argparse._SubParsersAction") -> None:
        """Register runtime-specific subcommand parsers.

        Override to add subcommands that only apply to this runtime.
        Called from configure_subcommands() after universal commands are registered.
        """
        pass

    @property
    def health_check_timeout(self) -> int:
        """Seconds to wait for the runtime server to become healthy."""
        return 20

    def is_healthy(self, conn: HTTPConnection, args: Any, model_name: str | None = None) -> bool:
        """Check server health. Override to implement runtime-specific health checks."""
        raise NotImplementedError(f"Runtime plugin '{self.name}' does not implement is_healthy()")
