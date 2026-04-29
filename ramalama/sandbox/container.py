from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, cast

from ramalama.common import perror
from ramalama.config import ActiveConfig
from ramalama.engine import Engine, stop_container
from ramalama.plugins.loader import get_runtime
from ramalama.transports.base import compute_serving_port
from ramalama.transports.transport_factory import New

if TYPE_CHECKING:
    from ramalama.sandbox.agent import Agent, SandboxEngineArgsType


class SandboxEngine(Engine):
    """Engine for running sandbox containers."""

    def __init__(self, args: SandboxEngineArgsType) -> None:
        super().__init__(args)

    def base_args(self) -> None:
        self.add_args("run", "--rm", "-i")

    def is_tty_cmd(self) -> bool:
        return getattr(self.args, "subcommand", "") == "sandbox"

    def add_network(self) -> None:
        self.add_args(f"--network=container:{self.args.name}")  # type: ignore[attr-defined]

    def add_workdir(self, args: SandboxEngineArgsType):
        if args.workdir:
            self.add_volume(args.workdir, "/work", opts="rw")
            self.add_args("--workdir=/work")

    def add_port_option(self) -> None:
        pass

    def add_oci_runtime(self) -> None:
        pass

    def add_detach_option(self) -> None:
        pass

    def add_device_options(self) -> None:
        pass


def _resolve_sandbox_engine(args: SandboxEngineArgsType) -> str:
    """Resolve which sandbox engine to use via explicit flag or detection cascade."""
    engine = getattr(args, "sandbox_engine", None)

    if engine is not None:
        return engine

    # Detection cascade: openshell → podman → docker
    try:
        from ramalama.sandbox.openshell import openshell_available

        if openshell_available():
            return "openshell"
    except Exception:
        pass

    # Fall back to whatever container engine is configured
    if args.container:
        return "podman" if args.engine and "podman" in str(args.engine) else "docker"

    raise ValueError("ramalama sandbox requires a container engine or OpenShell")


def run_sandbox(args: SandboxEngineArgsType, agent_cls: type[Agent]):
    """Orchestrate model server and sandbox containers."""

    resolved = _resolve_sandbox_engine(args)

    if resolved == "openshell":
        # Check for known incompatibilities that require container fallback
        use_openshell = True
        explicit = getattr(args, "sandbox_engine", None) == "openshell"

        if args.workdir:
            if explicit:
                raise ValueError(
                    "--workdir is not supported with --sandbox-engine openshell (host directories cannot be mounted)"
                )
            perror("--workdir is not supported with OpenShell sandboxes; falling back to container engine")
            use_openshell = False

        if use_openshell and not args.ARGS and sys.stdin.isatty():
            if explicit:
                raise ValueError("interactive TTY sessions are not supported with --sandbox-engine openshell")
            perror("Interactive TTY sessions are not supported with OpenShell; falling back to container engine")
            use_openshell = False

        if use_openshell:
            from ramalama.sandbox.openshell import run_sandbox_openshell

            return run_sandbox_openshell(args, agent_cls)

        # Fall through to container path
        resolved = "podman" if args.container and args.engine and "podman" in str(args.engine) else "docker"

    if resolved in ("podman", "docker"):
        if not args.container:
            raise ValueError("ramalama sandbox requires a container engine")
    else:
        raise ValueError(f"unknown sandbox engine: {resolved}")

    args.port = compute_serving_port(args)

    model = New(args.MODEL, args)
    model.ensure_model_exists(args)

    runtime = get_runtime(ActiveConfig().runtime)
    cmd = runtime.handle_subcommand("serve", cast(argparse.Namespace, args))

    model.serve_nonblocking(args, cmd)

    agent = agent_cls(args, model.model_alias)

    if args.dryrun:
        agent.engine.dryrun()
        return

    try:
        # Wait for model server to be healthy
        model.wait_for_healthy(args)  # type: ignore[union-attr]

        # Launch agent
        agent.run()
    finally:
        args.ignore = True  # type: ignore[attr-defined]
        stop_container(args, args.name, remove=True)  # type: ignore[attr-defined]
