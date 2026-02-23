from http.client import HTTPConnection
from typing import TYPE_CHECKING, Any

from ramalama.common import ContainerEntryPoint
from ramalama.logger import logger
from ramalama.plugins.runtimes.common import ContainerizedInferenceRuntimePlugin

if TYPE_CHECKING:
    from ramalama.command.context import RamalamaCommandContext


class VllmPlugin(ContainerizedInferenceRuntimePlugin):
    @property
    def name(self) -> str:
        return "vllm"

    def build_command(self, command: str, ctx: "RamalamaCommandContext") -> list[str]:
        if command in ("serve", "run"):
            return self._serve_run_cmd(ctx)
        if command == "rag":
            return self._rag_generate_cmd(ctx)
        if command == "convert":
            return self._convert_cmd(ctx)
        raise NotImplementedError(f"vllm plugin does not implement command '{command}'")

    def _serve_run_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        cmd: list[str] = []

        if ctx.host.is_container:
            cmd.append(ContainerEntryPoint())
        else:
            cmd += ["python3", "-m", "vllm.entrypoints.openai.api_server"]

        if ctx.model is not None:
            cmd += ["--model", ctx.model.model_path]
            cmd += ["--served-model-name", ctx.model.alias]

        max_model_len = ctx.args.ctx_size if ctx.args.ctx_size else 2048  # type: ignore[union-attr]
        cmd += ["--max_model_len", str(max_model_len)]

        if ctx.args.port is not None:  # type: ignore[union-attr]
            cmd += ["--port", str(ctx.args.port)]  # type: ignore[union-attr]

        if ctx.args.seed is not None:  # type: ignore[union-attr]
            cmd += ["--seed", str(ctx.args.seed)]  # type: ignore[union-attr]

        if ctx.args.runtime_args:  # type: ignore[union-attr]
            cmd.extend(ctx.args.runtime_args)  # type: ignore[union-attr]

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

    def _convert_cmd(self, ctx: "RamalamaCommandContext") -> list[str]:
        model_name = ctx.model.model_name if ctx.model is not None else ""  # type: ignore[union-attr]
        return [
            "convert_hf_to_gguf.py",
            "--outfile",
            f"/output/{model_name}.gguf",
            "/model",
        ]

    def get_container_image(self, config: Any, gpu_type: str) -> str | None:
        image = None
        if gpu_type:
            image = config.images.get(f"VLLM_{gpu_type}")
        if not image:
            image = config.images.get("VLLM", "docker.io/vllm/vllm-openai")
        return image if ":" in image else f"{image}:latest"

    @property
    def health_check_timeout(self) -> int:
        return 180

    def is_healthy(self, conn: HTTPConnection, args: Any, model_name: str | None = None) -> bool:
        conn.request("GET", "/ping")
        resp = conn.getresponse()
        if resp.status != 200:
            logger.debug(f"Container {args.name} /ping status code: {resp.status}: {resp.reason}")
            return False
        logger.debug(f"Container {args.name} is healthy")
        return True
