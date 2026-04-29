from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from typing import Optional

from ramalama.arg_types import BaseEngineArgsType
from ramalama.common import run_cmd
from ramalama.config import ActiveConfig
from ramalama.plugins.loader import get_runtime
from ramalama.sandbox.container import SandboxEngine


def _add_common_sandbox_args(parser: argparse.ArgumentParser) -> None:
    """Add --workdir, --sandbox-engine, and ARGS arguments shared by all sandbox subcommands."""
    parser.add_argument(
        "-w",
        "--workdir",
        help="local directory to mount into the sandbox container at /work",
    )
    parser.add_argument(
        "--sandbox-engine",
        choices=["openshell", "podman", "docker"],
        default=None,
        help="sandbox backend (default: auto-detect, preferring openshell > podman > docker)",
    )
    parser.add_argument(
        "ARGS",
        nargs="*",
        help="instructions for the sandbox to process non-interactively",
    )


def add_sandbox_subparsers(subparsers: argparse._SubParsersAction, img_comp: Callable, model_comp: Callable):
    """
    Add subparsers to the provided "subparser" object for each subcommand of
    "ramalama sandbox".
    "img_comp" and "model_comp" are completer functions for images and models, respectively.
    "func" is the function that will be called when the subcommand is run.
    """
    runtime = get_runtime(ActiveConfig().runtime)
    parser = subparsers.add_parser("goose", help="run Goose in a sandbox, backed by a local AI Model")
    if getattr(runtime, "_add_inference_args", None):
        # Consider adding this to the plugin interface for commands which need to run an
        # inference server
        runtime._add_inference_args(parser, "serve")  # type: ignore[attr-defined]
    parser.add_argument("MODEL", completer=model_comp)
    parser.add_argument(
        "--goose-image",
        default="ghcr.io/block/goose:1.28.0",
        completer=img_comp,
        help="Goose container image",
    )
    _add_common_sandbox_args(parser)
    parser.set_defaults(func=run_sandbox_goose)
    yield parser

    parser = subparsers.add_parser("opencode", help="run OpenCode in a sandbox, backed by a local AI Model")
    if getattr(runtime, "_add_inference_args", None):
        runtime._add_inference_args(parser, "serve")  # type: ignore[attr-defined]
    parser.add_argument("MODEL", completer=model_comp)
    parser.add_argument(
        "--opencode-image",
        default="ghcr.io/anomalyco/opencode:1.3.7",
        completer=img_comp,
        help="OpenCode container image",
    )
    _add_common_sandbox_args(parser)
    parser.set_defaults(func=run_sandbox_opencode)
    yield parser


class SandboxEngineArgsType(BaseEngineArgsType):
    ARGS: list[str]
    workdir: Optional[str]
    sandbox_engine: Optional[str]


class Agent:
    """
    Run an agent in a container.
    """

    def __init__(self, args: SandboxEngineArgsType, model_name: str):
        self.engine = SandboxEngine(args)
        self.model_name = model_name

    def run(self) -> None:
        run_cmd(self.engine.exec_args, stdout=None, stdin=None)

    def openshell_env(self) -> dict[str, str]:
        raise NotImplementedError

    def openshell_command(self) -> list[str]:
        raise NotImplementedError

    def openshell_image(self) -> str:
        raise NotImplementedError


class GooseArgsType(SandboxEngineArgsType):
    goose_image: str


class Goose(Agent):
    """
    Run Goose in a sandbox.
    Environment variables required by Goose will be set, and any workdir specified will be mounted into the container.
    If args are provided, they will be passed to Goose to process non-interactively. If there are no arguments and stdin
    is a tty, an interactive session will be started. Otherwise, instructions will be read from stdin.
    """

    def __init__(self, args: GooseArgsType, model_name: str) -> None:
        super().__init__(args, model_name)
        if self.engine.use_podman:
            if platform.system() != "Windows":
                self.engine.add_args("--uidmap=+1000:0")
        self.engine.add_name(f"goose-{args.name}")  # type: ignore[attr-defined]
        self.add_env_options(args)
        self.engine.add_workdir(args)
        self.engine.add_args(args.goose_image)
        if args.ARGS:
            self.engine.add_args("run", "-t", " ".join(args.ARGS))
        elif self.engine.use_tty():
            self.engine.add_args("session")
        else:
            self.engine.add_args("run", "-i", "-")

    def add_env_options(self, args: GooseArgsType) -> None:
        self.engine.add_env_option("GOOSE_PROVIDER=openai")
        self.engine.add_env_option(f"OPENAI_HOST=http://localhost:{args.port}")
        self.engine.add_env_option("OPENAI_API_KEY=ramalama")
        self.engine.add_env_option(f"GOOSE_MODEL={self.model_name}")
        self.engine.add_env_option("GOOSE_TELEMETRY_ENABLED=false")
        self.engine.add_env_option("GOOSE_CLI_SHOW_THINKING=true")

    def openshell_env(self) -> dict[str, str]:
        from ramalama.sandbox.openshell import OPENSHELL_HOST

        return {
            "GOOSE_PROVIDER": "openai",
            "OPENAI_HOST": f"http://{OPENSHELL_HOST}:{self.engine.args.port}",
            "OPENAI_API_KEY": "ramalama",
            "GOOSE_MODEL": self.model_name,
            "GOOSE_TELEMETRY_ENABLED": "false",
            "GOOSE_CLI_SHOW_THINKING": "true",
        }

    def openshell_command(self) -> list[str]:
        args = self.engine.args
        if args.ARGS:
            return ["goose", "run", "-t", " ".join(args.ARGS)]
        return ["goose", "run", "-i", "-"]

    def openshell_image(self) -> str:
        return self.engine.args.goose_image  # type: ignore[attr-defined]


class OpenCodeArgsType(SandboxEngineArgsType):
    opencode_image: str


class OpenCode(Agent):
    """
    Run OpenCode in a sandbox.
    Environment variables required by OpenCode will be set, and any workdir specified will be mounted into the
    container. If args are provided, they will be passed to OpenCode to process non-interactively. If there are
    no arguments and stdin is a tty, an interactive TUI session will be started. Otherwise, instructions will be
    read from stdin.
    """

    def __init__(self, args: OpenCodeArgsType, model_name: str) -> None:
        super().__init__(args, model_name)
        self.engine.add_name(f"opencode-{args.name}")  # type: ignore[attr-defined]
        self.add_env_options(args)
        self.engine.add_workdir(args)
        self.engine.add_args(args.opencode_image)
        if args.ARGS or not self.engine.use_tty():
            # Use the "run" command to process args from the command-line or stdin non-interatively
            self.engine.add_args("run", "--thinking=true")
            self.engine.add(args.ARGS)
        # Running on a tty with no arguments will start the TUI for an interactive session

    def add_env_options(self, args: OpenCodeArgsType) -> None:
        config = {
            "$schema": "https://opencode.ai/config.json",
            "model": f"ramalama/{self.model_name}",
            "provider": {
                "ramalama": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "RamaLama",
                    "options": {
                        "baseURL": f"http://localhost:{args.port}/v1",
                        "apiKey": "ramalama",
                    },
                    "models": {
                        self.model_name: {
                            "name": self.model_name,
                        },
                    },
                },
            },
        }
        self.engine.add_env_option(f"OPENCODE_CONFIG_CONTENT={json.dumps(config)}")

    def openshell_env(self) -> dict[str, str]:
        from ramalama.sandbox.openshell import OPENSHELL_HOST

        config = {
            "$schema": "https://opencode.ai/config.json",
            "model": f"ramalama/{self.model_name}",
            "provider": {
                "ramalama": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "RamaLama",
                    "options": {
                        "baseURL": f"http://{OPENSHELL_HOST}:{self.engine.args.port}/v1",
                        "apiKey": "ramalama",
                    },
                    "models": {
                        self.model_name: {
                            "name": self.model_name,
                        },
                    },
                },
            },
        }
        return {"OPENCODE_CONFIG_CONTENT": json.dumps(config)}

    def openshell_command(self) -> list[str]:
        args = self.engine.args
        cmd = ["opencode", "run", "--thinking=true"]
        if args.ARGS:
            cmd.extend(args.ARGS)
        return cmd

    def openshell_image(self) -> str:
        return self.engine.args.opencode_image  # type: ignore[attr-defined]


def run_sandbox_goose(args: GooseArgsType):
    from ramalama.sandbox.container import run_sandbox

    run_sandbox(args, Goose)


def run_sandbox_opencode(args: OpenCodeArgsType):
    from ramalama.sandbox.container import run_sandbox

    run_sandbox(args, OpenCode)
