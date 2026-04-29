"""OpenShell sandbox backend for RamaLama.

Imported lazily from container.py. If the openshell package is not installed,
the import succeeds but openshell_available() returns False.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, cast

from ramalama.common import perror
from ramalama.config import ActiveConfig
from ramalama.engine import stop_container
from ramalama.logger import logger
from ramalama.plugins.loader import get_runtime
from ramalama.transports.base import compute_serving_port
from ramalama.transports.transport_factory import New

if TYPE_CHECKING:
    from ramalama.sandbox.agent import Agent, SandboxEngineArgsType

OPENSHELL_HOST = "host.openshell.internal"

EXEC_TIMEOUT_SECONDS = 3600

try:
    import grpc
    from openshell import SandboxClient, SandboxError
    from openshell._proto import openshell_pb2, sandbox_pb2

    OPENSHELL_AVAILABLE = True
except ImportError:
    OPENSHELL_AVAILABLE = False


def openshell_available() -> bool:
    """Check if OpenShell SDK is importable and a gateway is reachable."""
    if not OPENSHELL_AVAILABLE:
        return False
    try:
        client = SandboxClient.from_active_cluster(timeout=5.0)
        client.health()
        client.close()
        return True
    except Exception:
        return False


def _build_sandbox_spec(
    image: str,
    env: dict[str, str],
    port: int,
) -> openshell_pb2.SandboxSpec:
    """Build an OpenShell SandboxSpec for a RamaLama agent."""
    policy = sandbox_pb2.SandboxPolicy(
        version=1,
        filesystem=sandbox_pb2.FilesystemPolicy(
            include_workdir=True,
            read_only=["/usr", "/lib", "/etc", "/app", "/proc", "/dev/urandom"],
            read_write=["/sandbox", "/tmp"],
        ),
        landlock=sandbox_pb2.LandlockPolicy(compatibility="best_effort"),
        process=sandbox_pb2.ProcessPolicy(run_as_user="sandbox", run_as_group="sandbox"),
        network_policies={
            "model_server": sandbox_pb2.NetworkPolicyRule(
                name="model_server",
                endpoints=[
                    sandbox_pb2.NetworkEndpoint(
                        host=OPENSHELL_HOST,
                        port=port,
                    ),
                ],
                binaries=[
                    sandbox_pb2.NetworkBinary(path="/**"),
                ],
            ),
        },
    )

    return openshell_pb2.SandboxSpec(
        template=openshell_pb2.SandboxTemplate(image=image),
        environment=env,
        policy=policy,
    )


def _dryrun_openshell(image: str, env: dict[str, str], port: int, command: list[str]) -> None:
    """Print what would happen in dryrun mode."""
    perror("[OpenShell sandbox dryrun]")
    perror(f"  image: {image}")
    for k, v in env.items():
        display_v = v if len(v) < 120 else v[:117] + "..."
        perror(f"  env: {k}={display_v}")
    perror(f"  network_policy: model_server -> {OPENSHELL_HOST}:{port}")
    perror(f"  command: {' '.join(command)}")


def run_sandbox_openshell(args: SandboxEngineArgsType, agent_cls: type[Agent]) -> None:
    """Run an agent in an OpenShell sandbox with reconnect-on-timeout."""
    if not OPENSHELL_AVAILABLE:
        raise ValueError("OpenShell SDK is not installed")

    args.port = compute_serving_port(args)

    model = New(args.MODEL, args)
    model.ensure_model_exists(args)

    runtime = get_runtime(ActiveConfig().runtime)
    cmd = runtime.handle_subcommand("serve", cast(argparse.Namespace, args))

    model.serve_nonblocking(args, cmd)

    agent = agent_cls(args, model.model_alias)

    image = agent.openshell_image()
    env = agent.openshell_env()
    command = agent.openshell_command()
    port = int(args.port)

    if args.dryrun:
        _dryrun_openshell(image, env, port, command)
        return

    spec = _build_sandbox_spec(image, env, port)

    client = None
    sandbox = None
    try:
        model.wait_for_healthy(args)  # type: ignore[union-attr]

        client = SandboxClient.from_active_cluster(timeout=EXEC_TIMEOUT_SECONDS + 10)
        sandbox = client.create(spec=spec)
        client.wait_ready(sandbox.name)

        stdin_data = None
        if not args.ARGS and not sys.stdin.isatty():
            stdin_data = sys.stdin.buffer.read()

        # Exec with reconnect-on-timeout loop
        while True:
            try:
                result = client.exec(
                    sandbox.id,
                    command,
                    stream_output=True,
                    stdin=stdin_data,
                    timeout_seconds=EXEC_TIMEOUT_SECONDS,
                )
                sys.exit(result.exit_code)
            except grpc.RpcError as exc:
                if isinstance(exc, grpc.Call) and exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.debug("OpenShell exec timed out, reconnecting...")
                    stdin_data = None
                    continue
                raise
    except SandboxError as e:
        perror(f"OpenShell sandbox error: {e}")
        sys.exit(1)
    finally:
        if client is not None and sandbox is not None:
            try:
                client.delete(sandbox.name)
                client.wait_deleted(sandbox.name, timeout_seconds=30.0)
            except Exception as e:
                logger.debug(f"Failed to delete OpenShell sandbox: {e}")
            client.close()
        args.ignore = True  # type: ignore[attr-defined]
        stop_container(args, args.name, remove=True)  # type: ignore[attr-defined]
