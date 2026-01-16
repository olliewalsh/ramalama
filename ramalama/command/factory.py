import argparse
import ast
import json
import shlex
from pathlib import Path
from typing import Any

import jinja2
import jsonschema
import yaml

from ramalama.command import context, error, schema
from ramalama.common import ContainerEntryPoint
from ramalama.config import CONFIG, get_inference_schema_files, get_inference_spec_files
from ramalama.logger import logger


def is_truthy(resolved_stmt: str) -> bool:
    return resolved_stmt not in ["None", "False", "", "[]", "{}"]


class CommandFactory:
    spec_files = get_inference_spec_files()
    schema_files = get_inference_schema_files()

    def __init__(self, runtime: str):
        self.runtime = runtime
        self.spec_file = self.spec_files.get(self.runtime, None)
        if self.spec_file is None:
            raise FileNotFoundError(f"No specification file found for runtime '{self.runtime}' ")

        spec_data = self._load_file(self.spec_file)
        if schema.VERSION_FIELD not in spec_data:
            raise error.InvalidInferenceEngineSpecError(
                str(self.spec_file), f"Missing required field '{schema.VERSION_FIELD}' "
            )
        try:
            self._validate_spec(spec_data)
        except Exception as ex:
            logger.debug(f"Inference engine specification validation failed for '{self.runtime}'", exc_info=True)
            raise error.InvalidInferenceEngineSpecError(str(self.spec_file), str(ex)) from ex

        self.spec_data = self._migrate(spec_data)
        logger.debug(f"Inference engine specification data for '{self.runtime}' loaded from '{self.spec_file}'")

    def __call__(self, command: str, ctx: context.RamalamaCommandContext) -> list[str]:
        spec = schema.CommandSpecV1.from_dict(self.spec_data, command)
        if spec is None:
            raise NotImplementedError(f"The specification for '{self.runtime}' does not implement command '{command}' ")

        cmd = self._resolve_cmd(spec, ctx)
        logger.debug(f"Resolved command for '{self.runtime}' command '{command}' to '{cmd}'")
        return cmd

    def _resolve_cmd(self, spec: schema.CommandSpecV1, ctx: context.RamalamaCommandContext) -> list[str]:
        engine = spec.command.engine

        cmd = []

        if not CONFIG.container:
            if CONFIG.runtime_config.native_binary is not None:
                cmd.append(CONFIG.runtime_config.native_binary)
            else:
                cmd.append(engine.native_cli.binary)
            if CONFIG.runtime_config.native_args is not None:
                cmd.extend(CONFIG.runtime_config.native_args)
            else:
                cmd.extend(engine.native_cli.args)
        else:
            if CONFIG.runtime_config.container_entrypoint is not None:
                cmd.append(ContainerEntryPoint(CONFIG.runtime_config.container_entrypoint))
            else:
                cmd.append(ContainerEntryPoint(engine.container_cli.entrypoint))
            if CONFIG.runtime_config.container_args is not None:
                cmd.extend(CONFIG.runtime_config.container_args)
            else:
                cmd.extend(engine.container_cli.args)

        for option in engine.options:
            should_add = option.condition is None or is_truthy(self._eval_stmt(option.condition, ctx))
            if not should_add:
                continue

            if option.value is None:
                cmd.append(option.name)
                continue

            value = self._eval_stmt(option.value, ctx)
            if is_truthy(value):
                if option.name:
                    cmd.append(option.name)

                if value.startswith("[") and value.endswith("]"):
                    cmd.extend(str(v) for v in ast.literal_eval(value))
                else:
                    cmd.append(str(value))

        return cmd

    def _eval_stmt(self, stmt: str, ctx: context.RamalamaCommandContext) -> Any:
        if not ("{{" in stmt and "}}" in stmt):
            return stmt

        return jinja2.Template(stmt).render(
            {
                "args": ctx.args,
                "model": ctx.model,
                "host": ctx.host,
            }
        )

    def _validate_spec(self, spec_data: dict):
        schema_version = spec_data[schema.VERSION_FIELD]
        schema_file = self.schema_files.get(schema_version.replace(".", "-"), None)
        if schema_file is None:
            raise FileNotFoundError(f"No schema file found for spec version '{schema_version}' ")
        schema_data = self._load_file(schema_file)
        jsonschema.validate(instance=spec_data, schema=schema_data)

    def _load_file(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"File '{path}' not found")

        with open(path, "r") as f:
            if path.suffix == ".json":
                return json.load(f)
            elif path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)

            raise NotImplementedError(f"File extension '{path.suffix}' not supported")

    def _migrate(self, spec_data: dict) -> dict:
        schema_version = spec_data[schema.VERSION_FIELD]
        if schema_version == "1.0.0":
            logger.debug(f"Migrating '{self.runtime}' spec data from version '{schema_version}' to '1.0.1'")
            for command in spec_data.get("commands", []):
                command["inference_engine"].setdefault("native_cli", {})["binary"] = command["inference_engine"][
                    "binary"
                ]
                # For v1.0.0 pass the binary as an argument to the container command
                command["inference_engine"].setdefault("container_cli", {})["entrypoint"] = command["inference_engine"][
                    "binary"
                ]
                del command["inference_engine"]["binary"]
            spec_data[schema.VERSION_FIELD] = "1.0.1"
            try:
                self._validate_spec(spec_data)
            except Exception as ex:
                raise error.InvalidInferenceEngineSpecError(str(self.spec_file), str(ex)) from ex
        return spec_data


def assemble_command(cli_args: argparse.Namespace) -> list[str]:
    runtime = str(cli_args.runtime)
    command = str(cli_args.subcommand)
    ctx = context.RamalamaCommandContext.from_argparse(cli_args)
    return CommandFactory(runtime)(command, ctx)
