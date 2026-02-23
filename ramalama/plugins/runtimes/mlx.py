import argparse
from typing import TYPE_CHECKING

from ramalama.logger import logger
from ramalama.plugins.runtimes.common import InferenceRuntimePlugin

if TYPE_CHECKING:
    from ramalama.command.context import RamalamaCommandContext


class MlxPlugin(InferenceRuntimePlugin):
    @property
    def name(self) -> str:
        return "mlx"

    def build_command(self, command: str, ctx: "RamalamaCommandContext") -> list[str]:
        if command in ("serve", "run"):
            return self._serve_run_cmd(ctx)
        raise NotImplementedError(f"mlx plugin does not implement command '{command}'")

    def _serve_run_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        cmd = ["mlx_lm.server"]

        if ctx.model is not None:
            cmd += ["--model", ctx.model.model_path]

        if ctx.args.temp is not None:  # type: ignore[union-attr]
            cmd += ["--temp", str(ctx.args.temp)]  # type: ignore[union-attr]

        if ctx.args.seed is not None:  # type: ignore[union-attr]
            cmd += ["--seed", str(ctx.args.seed)]  # type: ignore[union-attr]

        ctx_size = ctx.args.ctx_size if ctx.args.ctx_size else 0  # type: ignore[union-attr]
        if ctx_size > 0:
            cmd += ["--max-tokens", str(ctx_size)]

        if ctx.args.host is not None:  # type: ignore[union-attr]
            cmd += ["--host", str(ctx.args.host)]  # type: ignore[union-attr]

        if ctx.args.port is not None:  # type: ignore[union-attr]
            cmd += ["--port", str(ctx.args.port)]  # type: ignore[union-attr]

        if ctx.args.runtime_args:  # type: ignore[union-attr]
            cmd.extend(ctx.args.runtime_args)  # type: ignore[union-attr]

        return cmd

    def setup_args(self, args: argparse.Namespace) -> None:
        if getattr(args, "container", None) is True:
            logger.info("MLX runtime automatically uses --nocontainer mode")
        args.container = False
