"""Unit tests for runtime plugins (llama.cpp, vllm, mlx)."""

import argparse
from unittest.mock import MagicMock

import pytest

from ramalama.command.context import (
    RamalamaArgsContext,
    RamalamaCommandContext,
    RamalamaHostContext,
    RamalamaRagArgsContext,
    RamalamaRagGenArgsContext,
)
from ramalama.common import ContainerEntryPoint
from ramalama.plugins.runtimes.common import ContainerizedInferenceRuntimePlugin, InferenceRuntimePlugin
from ramalama.plugins.runtimes.llama_cpp import LlamaCppPlugin
from ramalama.plugins.runtimes.mlx import MlxPlugin
from ramalama.plugins.runtimes.vllm import VllmPlugin


def make_args(
    container=True,
    ngl=-1,
    threads=4,
    temp=0.8,
    seed=None,
    ctx_size=0,
    cache_reuse=256,
    max_tokens=0,
    port="8080",
    host="0.0.0.0",
    logfile=None,
    debug=False,
    webui="on",
    thinking=True,
    model_draft=None,
    runtime_args=None,
    gguf=None,
) -> RamalamaArgsContext:
    ctx = RamalamaArgsContext()
    ctx.container = container
    ctx.ngl = ngl
    ctx.threads = threads
    ctx.temp = temp
    ctx.seed = seed
    ctx.ctx_size = ctx_size
    ctx.cache_reuse = cache_reuse
    ctx.max_tokens = max_tokens
    ctx.port = port
    ctx.host = host
    ctx.logfile = logfile
    ctx.debug = debug
    ctx.webui = webui
    ctx.thinking = thinking
    ctx.model_draft = model_draft
    ctx.runtime_args = runtime_args or []
    ctx.gguf = gguf
    return ctx


def make_rag_gen_args(
    debug=False,
    format="qdrant",
    ocr=False,
    paths=None,
    urls=None,
    inputdir="/input",
) -> RamalamaRagGenArgsContext:
    ctx = RamalamaRagGenArgsContext()
    ctx.debug = debug
    ctx.format = format
    ctx.ocr = ocr
    ctx.paths = paths
    ctx.urls = urls
    ctx.inputdir = inputdir
    return ctx


def make_rag_args(
    debug=False,
    port="9090",
    model_host="host.containers.internal",
    model_port="8080",
) -> RamalamaRagArgsContext:
    ctx = RamalamaRagArgsContext()
    ctx.debug = debug
    ctx.port = port
    ctx.model_host = model_host
    ctx.model_port = model_port
    return ctx


def make_host(is_container=True, should_colorize=False, rpc_nodes=None) -> RamalamaHostContext:
    return RamalamaHostContext(
        is_container=is_container,
        uses_nvidia=False,
        uses_metal=False,
        should_colorize=should_colorize,
        rpc_nodes=rpc_nodes,
    )


def make_model(model_path="/mnt/models/model.file", alias="mymodel", mmproj_path=None, chat_template_path=None):
    model = MagicMock()
    model.model_path = model_path
    model.alias = alias
    model.mmproj_path = mmproj_path
    model.chat_template_path = chat_template_path
    model.draft_model_path = ""
    model.model_name = "mymodel"
    return model


def make_ctx(args=None, model=None, host=None) -> RamalamaCommandContext:
    if args is None:
        args = make_args()
    if host is None:
        host = make_host()
    return RamalamaCommandContext(args=args, model=model, host=host)


class TestLlamaCppPlugin:
    def setup_method(self):
        self.plugin = LlamaCppPlugin()

    def test_name(self):
        assert self.plugin.name == "llama.cpp"

    def test_serve_basic(self):
        model = make_model()
        ctx = make_ctx(args=make_args(container=True), model=model, host=make_host(is_container=True))
        cmd = self.plugin.build_command("serve", ctx)

        assert cmd[0] == "llama-server"
        assert "--host" in cmd
        assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
        assert "--port" in cmd
        assert cmd[cmd.index("--port") + 1] == "8080"
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "/mnt/models/model.file"
        assert "--no-warmup" in cmd
        assert "--jinja" in cmd
        assert "--alias" in cmd
        assert "-ngl" in cmd

    def test_serve_nocontainer_uses_configured_host(self):
        model = make_model()
        ctx = make_ctx(
            args=make_args(container=False, host="127.0.0.1"), model=model, host=make_host(is_container=False)
        )
        cmd = self.plugin.build_command("serve", ctx)

        assert "--host" in cmd
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1"

    def test_serve_with_mmproj(self):
        model = make_model(mmproj_path="/mnt/models/mmproj.file")
        ctx = make_ctx(model=model)
        cmd = self.plugin.build_command("serve", ctx)

        assert "--mmproj" in cmd
        assert "--no-jinja" in cmd
        assert "--jinja" not in cmd
        assert "--chat-template-file" not in cmd

    def test_serve_with_chat_template(self):
        model = make_model(chat_template_path="/mnt/models/chat_template.file")
        ctx = make_ctx(model=model)
        cmd = self.plugin.build_command("serve", ctx)

        assert "--chat-template-file" in cmd
        assert "--jinja" in cmd
        assert "--no-jinja" not in cmd

    def test_serve_thinking_disabled(self):
        ctx = make_ctx(args=make_args(thinking=False), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--reasoning-budget" in cmd
        assert cmd[cmd.index("--reasoning-budget") + 1] == "0"

    def test_serve_thinking_enabled(self):
        ctx = make_ctx(args=make_args(thinking=True), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--reasoning-budget" not in cmd

    def test_serve_ctx_size(self):
        ctx = make_ctx(args=make_args(ctx_size=4096), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--ctx-size" in cmd
        assert cmd[cmd.index("--ctx-size") + 1] == "4096"

    def test_serve_ctx_size_zero_not_added(self):
        ctx = make_ctx(args=make_args(ctx_size=0), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--ctx-size" not in cmd

    def test_serve_webui_off(self):
        ctx = make_ctx(args=make_args(webui="off"), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--no-webui" in cmd

    def test_serve_webui_on_not_added(self):
        ctx = make_ctx(args=make_args(webui="on"), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--no-webui" not in cmd

    def test_serve_ngl_positive(self):
        ctx = make_ctx(args=make_args(ngl=40), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "-ngl" in cmd
        assert cmd[cmd.index("-ngl") + 1] == "40"

    def test_serve_ngl_negative_uses_999(self):
        ctx = make_ctx(args=make_args(ngl=-1), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "-ngl" in cmd
        assert cmd[cmd.index("-ngl") + 1] == "999"

    def test_serve_max_tokens(self):
        ctx = make_ctx(args=make_args(max_tokens=512), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "-n" in cmd
        assert cmd[cmd.index("-n") + 1] == "512"

    def test_serve_max_tokens_zero_not_added(self):
        ctx = make_ctx(args=make_args(max_tokens=0), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "-n" not in cmd

    def test_serve_log_colors(self):
        ctx = make_ctx(model=make_model(), host=make_host(should_colorize=True))
        cmd = self.plugin.build_command("serve", ctx)

        assert "--log-colors" in cmd
        assert cmd[cmd.index("--log-colors") + 1] == "on"

    def test_serve_rpc_nodes(self):
        ctx = make_ctx(model=make_model(), host=make_host(rpc_nodes="192.168.1.1:50052"))
        cmd = self.plugin.build_command("serve", ctx)

        assert "--rpc" in cmd
        assert cmd[cmd.index("--rpc") + 1] == "192.168.1.1:50052"

    def test_serve_runtime_args(self):
        ctx = make_ctx(args=make_args(runtime_args=["--extra", "flag"]), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--extra" in cmd
        assert "flag" in cmd

    def test_serve_seed(self):
        ctx = make_ctx(args=make_args(seed=42), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--seed" in cmd
        assert cmd[cmd.index("--seed") + 1] == "42"

    def test_serve_debug(self):
        ctx = make_ctx(args=make_args(debug=True), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "-v" in cmd

    def test_run_same_as_serve(self):
        model = make_model()
        ctx = make_ctx(model=model)
        assert self.plugin.build_command("serve", ctx) == self.plugin.build_command("run", ctx)

    def test_perplexity(self):
        ctx = make_ctx(args=make_args(ngl=20, threads=8), model=make_model())
        cmd = self.plugin.build_command("perplexity", ctx)

        assert cmd[0] == "llama-perplexity"
        assert "--model" in cmd
        assert "-ngl" in cmd
        assert cmd[cmd.index("-ngl") + 1] == "20"
        assert "--threads" in cmd
        assert cmd[cmd.index("--threads") + 1] == "8"

    def test_bench(self):
        ctx = make_ctx(args=make_args(ngl=30), model=make_model())
        cmd = self.plugin.build_command("bench", ctx)

        assert cmd[0] == "llama-bench"
        assert "--model" in cmd
        assert "-ngl" in cmd
        assert "-o" in cmd
        assert cmd[cmd.index("-o") + 1] == "json"

    def test_rag_generate(self):
        args = make_rag_gen_args(format="qdrant", paths=["/some/path"], inputdir="/input")
        ctx = make_ctx(args=args)
        cmd = self.plugin.build_command("rag", ctx)

        assert cmd[0] == "doc2rag"
        assert "--format" in cmd
        assert cmd[cmd.index("--format") + 1] == "qdrant"
        assert "/output" in cmd
        assert "/input" in cmd

    def test_rag_generate_with_debug(self):
        args = make_rag_gen_args(debug=True)
        ctx = make_ctx(args=args)
        cmd = self.plugin.build_command("rag", ctx)

        assert "--debug" in cmd

    def test_rag_generate_with_ocr(self):
        args = make_rag_gen_args(ocr=True)
        ctx = make_ctx(args=args)
        cmd = self.plugin.build_command("rag", ctx)

        assert "--ocr" in cmd

    def test_rag_generate_with_urls(self):
        args = make_rag_gen_args(urls=["http://example.com", "http://other.com"])
        ctx = make_ctx(args=args)
        cmd = self.plugin.build_command("rag", ctx)

        assert "http://example.com" in cmd
        assert "http://other.com" in cmd

    def test_run_rag(self):
        args = make_rag_args(port="9090", model_host="host.containers.internal", model_port="8080")
        ctx = make_ctx(args=args)
        cmd = self.plugin.build_command("run --rag", ctx)

        assert cmd[0] == "rag_framework"
        assert "serve" in cmd
        assert "--port" in cmd
        assert cmd[cmd.index("--port") + 1] == "9090"
        assert "--model-host" in cmd
        assert cmd[cmd.index("--model-host") + 1] == "host.containers.internal"
        assert "--model-port" in cmd
        assert "/rag/vector.db" in cmd

    def test_serve_rag_same_as_run_rag(self):
        args = make_rag_args()
        ctx = make_ctx(args=args)
        assert self.plugin.build_command("run --rag", ctx) == self.plugin.build_command("serve --rag", ctx)

    def test_convert(self):
        model = make_model()
        ctx = make_ctx(model=model)
        cmd = self.plugin.build_command("convert", ctx)

        assert cmd[0] == "convert_hf_to_gguf.py"
        assert "--outfile" in cmd
        assert "/output/mymodel.gguf" in cmd
        assert "/model" in cmd

    def test_quantize(self):
        args = make_args()
        args.gguf = "Q4_K_M"
        model = make_model()
        ctx = make_ctx(args=args, model=model)
        cmd = self.plugin.build_command("quantize", ctx)

        assert cmd[0] == "llama-quantize"
        assert "/model/mymodel.gguf" in cmd
        assert "/model/mymodel-Q4_K_M.gguf" in cmd
        assert "Q4_K_M" in cmd

    def test_unsupported_command_raises(self):
        ctx = make_ctx()
        with pytest.raises(NotImplementedError):
            self.plugin.build_command("unknown_cmd", ctx)

    def test_no_container_image_override(self):
        config = MagicMock()
        assert self.plugin.get_container_image(config, "cuda") is None


class TestVllmPlugin:
    def setup_method(self):
        self.plugin = VllmPlugin()

    def test_name(self):
        assert self.plugin.name == "vllm"

    def test_serve_in_container(self):
        model = make_model()
        ctx = make_ctx(model=model, host=make_host(is_container=True))
        cmd = self.plugin.build_command("serve", ctx)

        assert isinstance(cmd[0], ContainerEntryPoint)
        assert "--model" in cmd
        assert "--served-model-name" in cmd
        assert "--max_model_len" in cmd
        assert "--port" in cmd

    def test_serve_nocontainer(self):
        model = make_model()
        ctx = make_ctx(model=model, host=make_host(is_container=False))
        cmd = self.plugin.build_command("serve", ctx)

        assert cmd[0] == "python3"
        assert cmd[1] == "-m"
        assert cmd[2] == "vllm.entrypoints.openai.api_server"

    def test_serve_default_max_model_len(self):
        ctx = make_ctx(args=make_args(ctx_size=0), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--max_model_len" in cmd
        assert cmd[cmd.index("--max_model_len") + 1] == "2048"

    def test_serve_custom_ctx_size(self):
        ctx = make_ctx(args=make_args(ctx_size=8192), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--max_model_len" in cmd
        assert cmd[cmd.index("--max_model_len") + 1] == "8192"

    def test_serve_seed(self):
        ctx = make_ctx(args=make_args(seed=123), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--seed" in cmd
        assert cmd[cmd.index("--seed") + 1] == "123"

    def test_serve_seed_not_added_when_none(self):
        ctx = make_ctx(args=make_args(seed=None), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--seed" not in cmd

    def test_serve_runtime_args(self):
        ctx = make_ctx(args=make_args(runtime_args=["--tensor-parallel-size", "2"]), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--tensor-parallel-size" in cmd
        assert "2" in cmd

    def test_run_same_as_serve(self):
        model = make_model()
        ctx = make_ctx(model=model)
        assert self.plugin.build_command("serve", ctx) == self.plugin.build_command("run", ctx)

    def test_get_container_image_with_gpu(self):
        config = MagicMock()
        config.images.get.side_effect = lambda key, default=None: {
            "VLLM_CUDA_VISIBLE_DEVICES": "docker.io/vllm/vllm-openai:cuda",
        }.get(key, default)

        image = self.plugin.get_container_image(config, "CUDA_VISIBLE_DEVICES")
        assert image == "docker.io/vllm/vllm-openai:cuda"

    def test_get_container_image_fallback(self):
        config = MagicMock()
        config.images.get.side_effect = lambda key, default=None: default

        image = self.plugin.get_container_image(config, "CUDA_VISIBLE_DEVICES")
        assert image == "docker.io/vllm/vllm-openai:latest"

    def test_get_container_image_no_gpu(self):
        config = MagicMock()
        config.images.get.side_effect = lambda key, default=None: default

        image = self.plugin.get_container_image(config, "")
        assert image == "docker.io/vllm/vllm-openai:latest"

    def test_get_container_image_with_tag_not_modified(self):
        config = MagicMock()
        config.images.get.side_effect = lambda key, default=None: "docker.io/vllm/vllm-openai:v0.5.0"

        image = self.plugin.get_container_image(config, "")
        assert image == "docker.io/vllm/vllm-openai:v0.5.0"

    def test_rag_generate(self):
        args = make_rag_gen_args(format="json")
        ctx = make_ctx(args=args)
        cmd = self.plugin.build_command("rag", ctx)

        assert cmd[0] == "doc2rag"
        assert "--format" in cmd
        assert "/output" in cmd

    def test_unsupported_command_raises(self):
        ctx = make_ctx()
        with pytest.raises(NotImplementedError):
            self.plugin.build_command("bench", ctx)


class TestMlxPlugin:
    def setup_method(self):
        self.plugin = MlxPlugin()

    def test_name(self):
        assert self.plugin.name == "mlx"

    def test_serve_basic(self):
        model = make_model()
        ctx = make_ctx(args=make_args(temp=0.7, port="8080", host="0.0.0.0"), model=model)
        cmd = self.plugin.build_command("serve", ctx)

        assert cmd[0] == "mlx_lm.server"
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "/mnt/models/model.file"
        assert "--temp" in cmd
        assert "--port" in cmd

    def test_serve_with_ctx_size(self):
        ctx = make_ctx(args=make_args(ctx_size=2048), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--max-tokens" in cmd
        assert cmd[cmd.index("--max-tokens") + 1] == "2048"

    def test_serve_ctx_size_zero_not_added(self):
        ctx = make_ctx(args=make_args(ctx_size=0), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--max-tokens" not in cmd

    def test_serve_seed(self):
        ctx = make_ctx(args=make_args(seed=99), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--seed" in cmd
        assert cmd[cmd.index("--seed") + 1] == "99"

    def test_serve_runtime_args(self):
        ctx = make_ctx(args=make_args(runtime_args=["--verbose"]), model=make_model())
        cmd = self.plugin.build_command("serve", ctx)

        assert "--verbose" in cmd

    def test_run_same_as_serve(self):
        model = make_model()
        ctx = make_ctx(model=model)
        assert self.plugin.build_command("serve", ctx) == self.plugin.build_command("run", ctx)

    def test_setup_args_forces_nocontainer(self):
        args = argparse.Namespace(container=True)
        self.plugin.setup_args(args)
        assert args.container is False

    def test_setup_args_keeps_nocontainer(self):
        args = argparse.Namespace(container=False)
        self.plugin.setup_args(args)
        assert args.container is False

    def test_no_container_image_override(self):
        config = MagicMock()
        assert self.plugin.get_container_image(config, "cuda") is None

    def test_unsupported_command_raises(self):
        ctx = make_ctx()
        with pytest.raises(NotImplementedError):
            self.plugin.build_command("bench", ctx)

    def test_is_inference_runtime_plugin_not_containerized(self):
        assert isinstance(self.plugin, InferenceRuntimePlugin)
        assert not isinstance(self.plugin, ContainerizedInferenceRuntimePlugin)


class TestPluginClassHierarchy:
    """Verify the class hierarchy for all runtime plugins."""

    def test_llama_cpp_is_containerized(self):
        assert isinstance(LlamaCppPlugin(), ContainerizedInferenceRuntimePlugin)

    def test_vllm_is_containerized(self):
        assert isinstance(VllmPlugin(), ContainerizedInferenceRuntimePlugin)

    def test_mlx_is_not_containerized(self):
        assert isinstance(MlxPlugin(), InferenceRuntimePlugin)
        assert not isinstance(MlxPlugin(), ContainerizedInferenceRuntimePlugin)


class TestConfigureSubcommandsFiltering:
    """Verify configure_subcommands() only registers subcommands for the selected runtime."""

    def _make_parser(self):
        from ramalama.cli import ArgumentParserWithDefaults

        return ArgumentParserWithDefaults()

    def _name_map(self, parser):
        return next(a for a in parser._actions if hasattr(a, "_name_parser_map"))._name_parser_map

    def test_mlx_runtime_excludes_rag(self, monkeypatch):
        from ramalama.cli import configure_subcommands
        from ramalama.config import get_config

        monkeypatch.setattr(get_config(), "runtime", "mlx")
        parser = self._make_parser()
        configure_subcommands(parser)
        name_map = self._name_map(parser)
        assert "rag" not in name_map
        assert "run" in name_map
        assert "serve" in name_map

    def test_llama_cpp_runtime_includes_rag_with_container(self, monkeypatch):
        from ramalama.cli import configure_subcommands
        from ramalama.config import get_config

        monkeypatch.setattr(get_config(), "runtime", "llama.cpp")
        monkeypatch.setattr(get_config(), "container", True)
        parser = self._make_parser()
        configure_subcommands(parser)
        name_map = self._name_map(parser)
        assert "rag" in name_map
        assert "run" in name_map
        assert "serve" in name_map

    def test_llama_cpp_runtime_excludes_rag_nocontainer(self, monkeypatch):
        from ramalama.cli import configure_subcommands
        from ramalama.config import get_config

        monkeypatch.setattr(get_config(), "runtime", "llama.cpp")
        monkeypatch.setattr(get_config(), "container", False)
        parser = self._make_parser()
        configure_subcommands(parser)
        name_map = self._name_map(parser)
        assert "rag" not in name_map
        assert "run" in name_map
        assert "serve" in name_map

    def test_vllm_runtime_includes_rag_with_container(self, monkeypatch):
        from ramalama.cli import configure_subcommands
        from ramalama.config import get_config

        monkeypatch.setattr(get_config(), "runtime", "vllm")
        monkeypatch.setattr(get_config(), "container", True)
        parser = self._make_parser()
        configure_subcommands(parser)
        name_map = self._name_map(parser)
        assert "rag" in name_map
        assert "run" in name_map
        assert "serve" in name_map

    def test_vllm_runtime_excludes_rag_nocontainer(self, monkeypatch):
        from ramalama.cli import configure_subcommands
        from ramalama.config import get_config

        monkeypatch.setattr(get_config(), "runtime", "vllm")
        monkeypatch.setattr(get_config(), "container", False)
        parser = self._make_parser()
        configure_subcommands(parser)
        name_map = self._name_map(parser)
        assert "rag" not in name_map
        assert "run" in name_map
        assert "serve" in name_map

    def test_unknown_runtime_raises(self, monkeypatch):
        from ramalama.cli import configure_subcommands
        from ramalama.config import get_config

        monkeypatch.setattr(get_config(), "runtime", "no-such-runtime")
        parser = self._make_parser()
        with pytest.raises(ValueError, match="Unknown runtime 'no-such-runtime'"):
            configure_subcommands(parser)
