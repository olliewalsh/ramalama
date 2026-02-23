import argparse
import json
from dataclasses import asdict
from http.client import HTTPConnection
from typing import TYPE_CHECKING, Any

from ramalama.logger import logger
from ramalama.plugins.runtimes.common import ContainerizedInferenceRuntimePlugin

if TYPE_CHECKING:
    from ramalama.command.context import RamalamaCommandContext


class LlamaCppPlugin(ContainerizedInferenceRuntimePlugin):
    @property
    def name(self) -> str:
        return "llama.cpp"

    def build_command(self, command: str, ctx: "RamalamaCommandContext") -> list[str]:
        if command in ("serve", "run"):
            return self._serve_run_cmd(ctx)
        if command == "perplexity":
            return self._perplexity_cmd(ctx)
        if command == "bench":
            return self._bench_cmd(ctx)
        if command == "rag":
            return self._rag_generate_cmd(ctx)
        if command in ("run --rag", "serve --rag"):
            return self._rag_framework_cmd(ctx)
        if command == "convert":
            return self._convert_cmd(ctx)
        if command == "quantize":
            return self._quantize_cmd(ctx)
        raise NotImplementedError(f"llama.cpp plugin does not implement command '{command}'")

    def _serve_run_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        cmd = ["llama-server"]

        # --host: use 0.0.0.0 in container, or the configured host otherwise
        host = '0.0.0.0' if ctx.host.is_container else ctx.args.host  # type: ignore[union-attr]
        if host is not None:
            cmd += ["--host", str(host)]

        if ctx.args.port is not None:  # type: ignore[union-attr]
            cmd += ["--port", str(ctx.args.port)]  # type: ignore[union-attr]

        if ctx.args.logfile:  # type: ignore[union-attr]
            cmd += ["--log-file", str(ctx.args.logfile)]  # type: ignore[union-attr]

        if ctx.model is not None:
            cmd += ["--model", ctx.model.model_path]

            if ctx.model.mmproj_path:
                cmd += ["--mmproj", str(ctx.model.mmproj_path)]

            if not ctx.model.mmproj_path and ctx.model.chat_template_path:
                cmd += ["--chat-template-file", str(ctx.model.chat_template_path)]

            if not ctx.model.mmproj_path:
                cmd.append("--jinja")
            else:
                cmd.append("--no-jinja")

        cmd.append("--no-warmup")

        if not ctx.args.thinking:  # type: ignore[union-attr]
            cmd += ["--reasoning-budget", "0"]

        if ctx.model is not None:
            cmd += ["--alias", ctx.model.alias]

        if ctx.args.ctx_size and ctx.args.ctx_size > 0:  # type: ignore[union-attr]
            cmd += ["--ctx-size", str(ctx.args.ctx_size)]  # type: ignore[union-attr]

        if ctx.args.temp is not None:  # type: ignore[union-attr]
            cmd += ["--temp", str(ctx.args.temp)]  # type: ignore[union-attr]

        if ctx.args.cache_reuse is not None:  # type: ignore[union-attr]
            cmd += ["--cache-reuse", str(ctx.args.cache_reuse)]  # type: ignore[union-attr]

        if ctx.args.debug:  # type: ignore[union-attr]
            cmd.append("-v")

        if ctx.args.webui == 'off':  # type: ignore[union-attr]
            cmd.append("--no-webui")

        ngl_val = 999 if ctx.args.ngl < 0 else ctx.args.ngl  # type: ignore[union-attr, operator]
        cmd += ["-ngl", str(ngl_val)]

        if ctx.args.model_draft:  # type: ignore[union-attr]
            draft_path = ctx.model.draft_model_path if ctx.model is not None else ""
            if draft_path:
                cmd += ["--model-draft", draft_path]
            cmd += ["-ngld", str(ngl_val)]

        if ctx.args.threads is not None:  # type: ignore[union-attr]
            cmd += ["--threads", str(ctx.args.threads)]  # type: ignore[union-attr]

        if ctx.args.seed is not None:  # type: ignore[union-attr]
            cmd += ["--seed", str(ctx.args.seed)]  # type: ignore[union-attr]

        if ctx.host.should_colorize:
            cmd += ["--log-colors", "on"]

        if ctx.host.rpc_nodes:
            cmd += ["--rpc", str(ctx.host.rpc_nodes)]

        if ctx.args.max_tokens and ctx.args.max_tokens > 0:  # type: ignore[union-attr]
            cmd += ["-n", str(ctx.args.max_tokens)]  # type: ignore[union-attr]

        if ctx.args.runtime_args:  # type: ignore[union-attr]
            cmd.extend(ctx.args.runtime_args)  # type: ignore[union-attr]

        return cmd

    def _perplexity_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        cmd = ["llama-perplexity"]

        if ctx.model is not None:
            cmd += ["--model", ctx.model.model_path]

        ngl_val = 999 if ctx.args.ngl < 0 else ctx.args.ngl  # type: ignore[union-attr, operator]
        cmd += ["-ngl", str(ngl_val)]

        if ctx.args.model_draft:  # type: ignore[union-attr]
            cmd += ["-ngld", str(ngl_val)]

        if ctx.args.threads is not None:  # type: ignore[union-attr]
            cmd += ["--threads", str(ctx.args.threads)]  # type: ignore[union-attr]

        return cmd

    def _bench_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        cmd = ["llama-bench"]

        if ctx.model is not None:
            cmd += ["--model", ctx.model.model_path]

        ngl_val = 999 if ctx.args.ngl < 0 else ctx.args.ngl  # type: ignore[union-attr, operator]
        cmd += ["-ngl", str(ngl_val)]

        if ctx.args.model_draft:  # type: ignore[union-attr]
            cmd += ["-ngld", str(ngl_val)]

        if ctx.args.threads is not None:  # type: ignore[union-attr]
            cmd += ["--threads", str(ctx.args.threads)]  # type: ignore[union-attr]

        cmd += ["-o", "json"]

        return cmd

    def _rag_generate_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        cmd = ["doc2rag"]

        if ctx.args.debug:  # type: ignore[union-attr]
            cmd.append("--debug")

        if ctx.args.format:  # type: ignore[union-attr]
            cmd += ["--format", str(ctx.args.format)]  # type: ignore[union-attr]

        if ctx.args.ocr:  # type: ignore[union-attr]
            cmd.append("--ocr")

        cmd.append("/output")

        if ctx.args.paths and ctx.args.inputdir:  # type: ignore[union-attr]
            cmd.append(str(ctx.args.inputdir))  # type: ignore[union-attr]

        if ctx.args.urls:  # type: ignore[union-attr]
            cmd.extend(ctx.args.urls)  # type: ignore[union-attr]

        return cmd

    def _rag_framework_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        cmd = ["rag_framework"]

        if ctx.args.debug:  # type: ignore[union-attr]
            cmd.append("--debug")

        cmd.append("serve")

        if ctx.args.port is not None:  # type: ignore[union-attr]
            cmd += ["--port", str(ctx.args.port)]  # type: ignore[union-attr]

        if ctx.args.model_host is not None:  # type: ignore[union-attr]
            cmd += ["--model-host", str(ctx.args.model_host)]  # type: ignore[union-attr]

        if ctx.args.model_port is not None:  # type: ignore[union-attr]
            cmd += ["--model-port", str(ctx.args.model_port)]  # type: ignore[union-attr]

        cmd.append("/rag/vector.db")

        return cmd

    def _convert_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        model_name = ctx.model.model_name if ctx.model is not None else ""  # type: ignore[union-attr]
        return [
            "convert_hf_to_gguf.py",
            "--outfile",
            f"/output/{model_name}.gguf",
            "/model",
        ]

    def _quantize_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        model_name = ctx.model.model_name if ctx.model is not None else ""  # type: ignore[union-attr]
        gguf = ctx.args.gguf if ctx.args.gguf else ""  # type: ignore[union-attr]
        return [
            "llama-quantize",
            f"/model/{model_name}.gguf",
            f"/model/{model_name}-{gguf}.gguf",
            str(gguf),
        ]

    def is_healthy(self, conn: HTTPConnection, args: Any, model_name: str | None = None) -> bool:
        conn.request("GET", "/health")
        health_resp = conn.getresponse()
        health_resp.read()
        if health_resp.status not in (200, 404):
            logger.debug(f"Container {args.name} /health status code: {health_resp.status}: {health_resp.reason}")
            return False

        conn.request("GET", "/models")
        models_resp = conn.getresponse()
        if models_resp.status != 200:
            logger.debug(f"Container {args.name} /models status code {models_resp.status}: {models_resp.reason}")
            return False

        content = models_resp.read()
        if not content:
            logger.debug(f"Container {args.name} /models returned an empty response")
            return False

        body = json.loads(content)
        if "models" not in body:
            logger.debug(f"Container {args.name} /models does not include a model list in the response")
            return False

        model_names = [m["name"] for m in body["models"]]
        if not model_name:
            # Inline import to avoid circular dependency
            from ramalama.transports.transport_factory import New

            model_name = New(args.MODEL, args).model_alias

        if not any(model_name in name for name in model_names):
            logger.debug(
                f'Container {args.name} /models does not include "{model_name}" in the model list: {model_names}'
            )
            return False

        logger.debug(f"Container {args.name} is healthy")
        return True

    # --- subcommand registration ---

    def register_subcommands(self, subparsers: "argparse._SubParsersAction") -> None:
        super().register_subcommands(subparsers)
        # Function-level imports avoid circular dependency: cli.py imports plugins at
        # module level, but register_subcommands() is only called after cli.py is fully
        # initialized (from configure_subcommands()), so these imports are always safe.
        from ramalama.cli import (
            OverrideDefaultAction,
            add_network_argument,
            default_image,
            default_rag_image,
            local_images,
            local_models,
            runtime_options,
        )

        # bench / benchmark
        bench_parser = subparsers.add_parser("bench", aliases=["benchmark"], help="benchmark specified AI Model")
        runtime_options(bench_parser, "bench")
        bench_parser.add_argument("MODEL", completer=local_models)
        bench_parser.add_argument(
            "--format",
            choices=["table", "json"],
            default="table",
            help="output format (table or json)",
        )
        bench_parser.set_defaults(func=self._bench_handler)

        # perplexity
        perplexity_parser = subparsers.add_parser("perplexity", help="calculate perplexity for specified AI Model")
        runtime_options(perplexity_parser, "perplexity")
        perplexity_parser.add_argument("MODEL", completer=local_models)
        perplexity_parser.set_defaults(func=self._perplexity_handler)

        # convert
        from typing import get_args

        from ramalama.config import GGUF_QUANTIZATION_MODES, get_config

        config = get_config()
        convert_parser = subparsers.add_parser(
            "convert",
            help="convert AI Model from local storage to OCI Image",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        convert_parser.add_argument("--carimage", default=config.carimage, help=argparse.SUPPRESS)
        convert_parser.add_argument(
            "--gguf",
            choices=get_args(GGUF_QUANTIZATION_MODES),
            nargs="?",
            const=config.gguf_quantization_mode,
            default=None,
            help=f"GGUF quantization format. If specified without value, {config.gguf_quantization_mode} is used.",
        )
        add_network_argument(convert_parser)
        convert_parser.add_argument(
            "--rag-image",
            default=default_rag_image(),
            help="Image to use for conversion to GGUF",
            action=OverrideDefaultAction,
            completer=local_images,
        )
        convert_parser.add_argument(
            "--image",
            default=default_image(),
            help="Image to use for quantization",
            action=OverrideDefaultAction,
            completer=local_images,
        )
        convert_parser.add_argument(
            "--pull",
            dest="pull",
            type=str,
            default=config.pull,
            choices=["always", "missing", "never", "newer"],
            help="pull image policy",
        )
        convert_parser.add_argument(
            "--type",
            default=config.convert_type,
            choices=["artifact", "car", "raw"],
            help="""\
type of OCI Model Image to push.

Model "artifact" stores the AI Model as an OCI Artifact.
Model "car" includes base image with the model stored in a /models subdir.
Model "raw" contains the model and a link file model.file to it stored at /.""",
        )
        convert_parser.add_argument("SOURCE")
        convert_parser.add_argument("TARGET")
        convert_parser.set_defaults(func=self._convert_handler)

        # benchmarks (manage stored results)
        from ramalama.config import get_config

        config = get_config()
        storage_folder = config.benchmarks.storage_folder
        epilog = f"Storage folder: {storage_folder}" if storage_folder else "Storage folder: not configured"
        benchmarks_parser = subparsers.add_parser(
            "benchmarks",
            help="manage and view benchmark results",
            epilog=epilog,
        )
        benchmarks_parser.set_defaults(func=lambda _: benchmarks_parser.print_help())
        benchmarks_subparsers = benchmarks_parser.add_subparsers(dest="benchmarks_command", metavar="[command]")
        benchmarks_list_parser = benchmarks_subparsers.add_parser("list", help="list benchmark results")
        benchmarks_list_parser.add_argument(
            "--limit", type=int, default=None, help="limit number of results to display"
        )
        benchmarks_list_parser.add_argument("--offset", type=int, default=0, help="offset for pagination")
        benchmarks_list_parser.add_argument(
            "--format", choices=["table", "json"], default="table", help="output format (table or json)"
        )
        benchmarks_list_parser.set_defaults(func=self._benchmarks_list_handler)

    def _convert_handler(self, args: argparse.Namespace) -> None:
        from ramalama.cli import _get_source_model, get_shortnames
        from ramalama.transports.transport_factory import TransportFactory

        if not args.container:
            raise ValueError("convert command cannot be run with the --nocontainer option.")

        shortnames = get_shortnames()
        tgt = shortnames.resolve(args.TARGET)
        model = TransportFactory(tgt, args).create_oci()
        source_model = _get_source_model(args)
        model.convert(source_model, args)

    def _bench_handler(self, args: argparse.Namespace) -> None:
        from ramalama.command.factory import assemble_command
        from ramalama.transports.transport_factory import New

        model = New(args.MODEL, args)
        model.ensure_model_exists(args)
        model.bench(args, assemble_command(args))

    def _perplexity_handler(self, args: argparse.Namespace) -> None:
        from ramalama.command.factory import assemble_command
        from ramalama.transports.transport_factory import New

        model = New(args.MODEL, args)
        model.ensure_model_exists(args)
        model.perplexity(args, assemble_command(args))

    def _benchmarks_list_handler(self, args: argparse.Namespace) -> None:
        from ramalama.benchmarks.manager import BenchmarksManager
        from ramalama.benchmarks.utilities import print_bench_results
        from ramalama.config import get_config

        config = get_config()
        bench_manager = BenchmarksManager(config.benchmarks.storage_folder)
        results = bench_manager.list()

        if not results:
            print("No benchmark results found")
            return

        if args.format == "json":
            output = [asdict(item) for item in results]
            print(json.dumps(output, indent=2, sort_keys=True))
        else:
            print_bench_results(results)
