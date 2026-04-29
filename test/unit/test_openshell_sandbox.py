import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ramalama.cli import parse_args_from_cmd
from ramalama.sandbox import Goose, OpenCode
from ramalama.sandbox.openshell import OPENSHELL_HOST

TEST_MODEL = "qwen3:4b"


def _make_goose_args(engine="podman", **overrides):
    defaults = dict(
        engine=engine,
        dryrun=False,
        quiet=True,
        goose_image="ghcr.io/block/goose:latest",
        name="ramalama_model_abc",
        port="8080",
        thinking=False,
        workdir=None,
        subcommand="sandbox",
        ARGS=[],
        sandbox_engine=None,
        MODEL="test-model",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_opencode_args(engine="podman", **overrides):
    defaults = dict(
        engine=engine,
        dryrun=False,
        quiet=True,
        opencode_image="ghcr.io/anomalyco/opencode:latest",
        name="ramalama_model_abc",
        port="8080",
        thinking=False,
        workdir=None,
        subcommand="sandbox",
        ARGS=[],
        sandbox_engine=None,
        MODEL="test-model",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# --- CLI parsing tests ---


@pytest.mark.parametrize("agent", ["goose", "opencode"])
def test_sandbox_engine_default_none(agent):
    """--sandbox-engine should default to None (auto-detect)."""
    _, args = parse_args_from_cmd(["sandbox", agent, TEST_MODEL])
    assert args.sandbox_engine is None


@pytest.mark.parametrize("choice", ["openshell", "podman", "docker"])
@pytest.mark.parametrize("agent", ["goose", "opencode"])
def test_sandbox_engine_choices(agent, choice):
    """--sandbox-engine should accept openshell, podman, docker."""
    _, args = parse_args_from_cmd(["sandbox", agent, "--sandbox-engine", choice, TEST_MODEL])
    assert args.sandbox_engine == choice


@pytest.mark.parametrize("agent", ["goose", "opencode"])
def test_sandbox_engine_invalid_choice(agent):
    """--sandbox-engine should reject invalid choices."""
    with pytest.raises(SystemExit):
        parse_args_from_cmd(["sandbox", agent, "--sandbox-engine", "invalid", TEST_MODEL])


# --- Goose OpenShell method tests ---


def test_goose_openshell_env():
    """Goose.openshell_env() should use host.openshell.internal for the model URL."""
    goose = Goose(_make_goose_args(), "test-model")
    env = goose.openshell_env()
    assert env["GOOSE_PROVIDER"] == "openai"
    assert env["OPENAI_HOST"] == f"http://{OPENSHELL_HOST}:8080"
    assert env["OPENAI_API_KEY"] == "ramalama"
    assert env["GOOSE_MODEL"] == "test-model"
    assert env["GOOSE_TELEMETRY_ENABLED"] == "false"
    assert env["GOOSE_CLI_SHOW_THINKING"] == "true"


def test_goose_openshell_command_with_args():
    """Goose.openshell_command() with ARGS should produce 'run -t <message>'."""
    goose = Goose(_make_goose_args(ARGS=["hello", "ramalama"]), "test-model")
    assert goose.openshell_command() == ["goose", "run", "-t", "hello ramalama"]


def test_goose_openshell_command_no_args():
    """Goose.openshell_command() without ARGS should produce 'run -i -'."""
    goose = Goose(_make_goose_args(), "test-model")
    assert goose.openshell_command() == ["goose", "run", "-i", "-"]


def test_goose_openshell_image():
    """Goose.openshell_image() should return the goose image."""
    goose = Goose(_make_goose_args(goose_image="ghcr.io/block/goose:1.28.0"), "test-model")
    assert goose.openshell_image() == "ghcr.io/block/goose:1.28.0"


# --- OpenCode OpenShell method tests ---


def test_opencode_openshell_env():
    """OpenCode.openshell_env() should use host.openshell.internal in the config."""
    opencode = OpenCode(_make_opencode_args(), "test-model")
    env = opencode.openshell_env()
    assert "OPENCODE_CONFIG_CONTENT" in env
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["model"] == "ramalama/test-model"
    assert config["provider"]["ramalama"]["options"]["baseURL"] == f"http://{OPENSHELL_HOST}:8080/v1"
    assert config["provider"]["ramalama"]["options"]["apiKey"] == "ramalama"


def test_opencode_openshell_command_with_args():
    """OpenCode.openshell_command() with ARGS should produce 'run --thinking=true <args>'."""
    opencode = OpenCode(_make_opencode_args(ARGS=["hello", "ramalama"]), "test-model")
    assert opencode.openshell_command() == ["opencode", "run", "--thinking=true", "hello", "ramalama"]


def test_opencode_openshell_command_no_args():
    """OpenCode.openshell_command() without ARGS should produce 'run --thinking=true'."""
    opencode = OpenCode(_make_opencode_args(), "test-model")
    assert opencode.openshell_command() == ["opencode", "run", "--thinking=true"]


def test_opencode_openshell_image():
    """OpenCode.openshell_image() should return the opencode image."""
    opencode = OpenCode(_make_opencode_args(opencode_image="ghcr.io/anomalyco/opencode:1.3.7"), "test-model")
    assert opencode.openshell_image() == "ghcr.io/anomalyco/opencode:1.3.7"


# --- Detection cascade tests ---


def test_resolve_sandbox_engine_explicit():
    """Explicit --sandbox-engine should be returned directly."""
    from ramalama.sandbox.container import _resolve_sandbox_engine

    args = _make_goose_args(sandbox_engine="podman", container=True)
    assert _resolve_sandbox_engine(args) == "podman"


def test_resolve_sandbox_engine_explicit_openshell():
    """Explicit --sandbox-engine=openshell should be returned directly."""
    from ramalama.sandbox.container import _resolve_sandbox_engine

    args = _make_goose_args(sandbox_engine="openshell", container=True)
    assert _resolve_sandbox_engine(args) == "openshell"


def test_resolve_sandbox_engine_auto_openshell():
    """Auto-detect should prefer openshell when available."""
    from ramalama.sandbox.container import _resolve_sandbox_engine

    with patch("ramalama.sandbox.openshell.openshell_available", return_value=True):
        args = _make_goose_args(sandbox_engine=None, container=True)
        assert _resolve_sandbox_engine(args) == "openshell"


def test_resolve_sandbox_engine_auto_podman():
    """Auto-detect should fall back to podman when openshell unavailable."""
    from ramalama.sandbox.container import _resolve_sandbox_engine

    with patch("ramalama.sandbox.openshell.openshell_available", return_value=False):
        args = _make_goose_args(sandbox_engine=None, container=True, engine="podman")
        assert _resolve_sandbox_engine(args) == "podman"


def test_resolve_sandbox_engine_auto_docker():
    """Auto-detect should fall back to docker when openshell unavailable and engine is docker."""
    from ramalama.sandbox.container import _resolve_sandbox_engine

    with patch("ramalama.sandbox.openshell.openshell_available", return_value=False):
        args = _make_goose_args(sandbox_engine=None, container=True, engine="docker")
        assert _resolve_sandbox_engine(args) == "docker"


def test_resolve_sandbox_engine_no_engine():
    """Auto-detect should raise when nothing is available."""
    from ramalama.sandbox.container import _resolve_sandbox_engine

    with patch("ramalama.sandbox.openshell.openshell_available", return_value=False):
        args = _make_goose_args(sandbox_engine=None, container=False, engine=None)
        with pytest.raises(ValueError, match="requires a container engine or OpenShell"):
            _resolve_sandbox_engine(args)


# --- Fallback tests ---


def test_run_sandbox_workdir_fallback(monkeypatch, capsys):
    """OpenShell should fall back to container when --workdir is set."""
    from ramalama.sandbox.container import run_sandbox

    with (
        patch("ramalama.sandbox.openshell.openshell_available", return_value=True),
        patch("ramalama.sandbox.container.compute_serving_port", return_value="8080"),
        patch("ramalama.sandbox.container.New") as mock_new,
        patch("ramalama.sandbox.container.get_runtime") as mock_runtime,
    ):
        mock_model = MagicMock()
        mock_model.model_alias = "test-model"
        mock_new.return_value = mock_model
        mock_runtime.return_value.handle_subcommand.return_value = ["serve"]

        args = _make_goose_args(
            sandbox_engine=None,
            container=True,
            workdir="/tmp/myproject",
            dryrun=True,
        )
        run_sandbox(args, Goose)

        captured = capsys.readouterr()
        assert "falling back to container engine" in captured.err


def test_run_sandbox_tty_fallback(monkeypatch, capsys):
    """OpenShell should fall back to container for interactive TTY sessions."""
    from ramalama.sandbox.container import run_sandbox

    monkeypatch.setattr("ramalama.sandbox.container.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("ramalama.engine.sys.stdin.isatty", lambda: True)

    with (
        patch("ramalama.sandbox.openshell.openshell_available", return_value=True),
        patch("ramalama.sandbox.container.compute_serving_port", return_value="8080"),
        patch("ramalama.sandbox.container.New") as mock_new,
        patch("ramalama.sandbox.container.get_runtime") as mock_runtime,
    ):
        mock_model = MagicMock()
        mock_model.model_alias = "test-model"
        mock_new.return_value = mock_model
        mock_runtime.return_value.handle_subcommand.return_value = ["serve"]

        args = _make_goose_args(
            sandbox_engine=None,
            container=True,
            ARGS=[],
            dryrun=True,
        )
        run_sandbox(args, Goose)

        captured = capsys.readouterr()
        assert "falling back to container engine" in captured.err


def test_run_sandbox_explicit_openshell_workdir_error():
    """Explicit --sandbox-engine=openshell with --workdir should raise."""
    from ramalama.sandbox.container import run_sandbox

    args = _make_goose_args(
        sandbox_engine="openshell",
        container=True,
        workdir="/tmp/myproject",
    )
    with pytest.raises(ValueError, match="--workdir is not supported"):
        run_sandbox(args, Goose)


def test_run_sandbox_explicit_openshell_tty_error(monkeypatch):
    """Explicit --sandbox-engine=openshell with TTY and no ARGS should raise."""
    from ramalama.sandbox.container import run_sandbox

    monkeypatch.setattr("ramalama.sandbox.container.sys.stdin.isatty", lambda: True)

    args = _make_goose_args(
        sandbox_engine="openshell",
        container=True,
        ARGS=[],
    )
    with pytest.raises(ValueError, match="interactive TTY sessions are not supported"):
        run_sandbox(args, Goose)


# --- Dryrun test ---


def test_dryrun_openshell_output(capsys):
    """_dryrun_openshell should print image, env, policy, and command."""
    from ramalama.sandbox.openshell import _dryrun_openshell

    _dryrun_openshell(
        image="ghcr.io/block/goose:1.28.0",
        env={"GOOSE_PROVIDER": "openai", "OPENAI_HOST": "http://host.openshell.internal:8080"},
        port=8080,
        command=["goose", "run", "-t", "hello"],
    )
    captured = capsys.readouterr()
    assert "[OpenShell sandbox dryrun]" in captured.err
    assert "ghcr.io/block/goose:1.28.0" in captured.err
    assert "GOOSE_PROVIDER=openai" in captured.err
    assert "host.openshell.internal:8080" in captured.err
    assert "goose run -t hello" in captured.err


# --- Spec construction test ---


@pytest.mark.skipif(
    "openshell" not in sys.modules and not __import__("importlib").util.find_spec("openshell"),
    reason="openshell SDK not installed",
)
def test_build_sandbox_spec():
    """_build_sandbox_spec should produce a valid SandboxSpec."""
    from ramalama.sandbox.openshell import _build_sandbox_spec

    spec = _build_sandbox_spec(
        image="ghcr.io/block/goose:1.28.0",
        env={"GOOSE_PROVIDER": "openai"},
        port=8080,
    )
    assert spec.template.image == "ghcr.io/block/goose:1.28.0"
    assert spec.environment["GOOSE_PROVIDER"] == "openai"
    assert "model_server" in spec.policy.network_policies
    rule = spec.policy.network_policies["model_server"]
    assert rule.endpoints[0].host == OPENSHELL_HOST
    assert rule.endpoints[0].port == 8080
