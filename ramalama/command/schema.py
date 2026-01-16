from typing import Optional

VERSION_FIELD = "schema_version"


class CommandSpecV1:

    class Option:
        name: str
        description: Optional[str]
        value: Optional[str]
        required: bool
        condition: Optional[str]

        @staticmethod
        def from_dict(d: dict) -> "CommandSpecV1.Option":
            option = CommandSpecV1.Option()
            option.name = d["name"]
            option.description = d.get("description", None)
            option.value = d.get("value", None)
            option.required = d.get("required", True)
            option.condition = d.get("if", None)

            return option

    class NativeCli:
        binary: str
        args: list[str]

        @staticmethod
        def from_dict(d: dict) -> "CommandSpecV1.NativeCli":
            native_args = CommandSpecV1.NativeCli()
            native_args.binary = d["binary"]
            native_args.args = d.get("args", [])
            return native_args

    class ContainerCli:
        entrypoint: str
        args: list[str]

        @staticmethod
        def from_dict(d: dict) -> "CommandSpecV1.ContainerCli":
            container_args = CommandSpecV1.ContainerCli()
            # Note: entrypoint == None means use the container entrypoint
            container_args.entrypoint = d.get("entrypoint", None)
            container_args.args = d.get("args", [])
            return container_args

    class Engine:
        name: str
        native_cli: "CommandSpecV1.NativeCli"
        container_cli: "CommandSpecV1.ContainerCli"
        options: list["CommandSpecV1.Option"]

        @staticmethod
        def from_dict(d: dict) -> "CommandSpecV1.Engine":
            engine = CommandSpecV1.Engine()

            engine.name = d["name"]
            engine.native_cli = CommandSpecV1.NativeCli.from_dict(d.get("native_cli", {}))
            engine.container_cli = CommandSpecV1.ContainerCli.from_dict(d.get("container_cli", {}))
            engine.options = [CommandSpecV1.Option.from_dict(o) for o in d.get("options", [])]

            return engine

    class Command:
        name: str
        engine: "CommandSpecV1.Engine"

        @staticmethod
        def from_dict(d: dict) -> "CommandSpecV1.Command":
            command = CommandSpecV1.Command()
            command.name = d["name"]
            command.engine = CommandSpecV1.Engine.from_dict(d["inference_engine"])
            return command

    def __init__(self, command: "CommandSpecV1.Command"):
        self.command = command

    @staticmethod
    def from_dict(d: dict, command: str) -> Optional["CommandSpecV1"]:
        for cmd in d.get("commands", []):
            if cmd["name"] == command:
                return CommandSpecV1(CommandSpecV1.Command.from_dict(cmd))

        return None
