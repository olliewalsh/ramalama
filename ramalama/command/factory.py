import argparse

from ramalama.command import context
from ramalama.plugins.loader import get_runtime


def assemble_command(cli_args: argparse.Namespace) -> list[str]:
    runtime = str(cli_args.runtime)
    command = str(cli_args.subcommand)
    plugin = get_runtime(runtime)
    if plugin is None:
        raise RuntimeError(f"No runtime plugin found for runtime '{runtime}'")
    ctx = context.RamalamaCommandContext.from_argparse(cli_args)
    return plugin.build_command(command, ctx)
