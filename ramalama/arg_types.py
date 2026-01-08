from dataclasses import make_dataclass
from typing import List, Protocol, get_type_hints, Optional, Union

from ramalama.config import COLOR_OPTIONS, SUPPORTED_ENGINES, SUPPORTED_RUNTIMES, PathStr


def protocol_to_dataclass(proto_cls):
    hints = get_type_hints(proto_cls)
    fields = [(name, typ) for name, typ in hints.items()]
    return make_dataclass(f"{proto_cls.__name__}DC", fields)


class EngineArgType(Protocol):
    engine: Optional[SUPPORTED_ENGINES]


EngineArgs = protocol_to_dataclass(EngineArgType)


class ContainerArgType(Protocol):
    container: Optional[bool]


ContainerArgs = protocol_to_dataclass(ContainerArgType)


class StoreArgType(Protocol):
    engine: Optional[SUPPORTED_ENGINES]
    container: bool
    store: str


StoreArgs = protocol_to_dataclass(StoreArgType)


class BaseEngineArgsType(Protocol):
    """Arguments required by BaseEngine.__init__"""

    # Required attributes (accessed directly)
    engine: SUPPORTED_ENGINES
    dryrun: bool
    quiet: bool
    image: str
    # Optional attributes (accessed via getattr)
    pull: Optional[str]
    network: Optional[str]
    oci_runtime: Optional[str]
    selinux: Optional[bool]
    nocapdrop: Optional[bool]
    device: Optional[list[str]]
    podman_keep_groups: Optional[bool]
    # Optional attributes for labels
    MODEL: Optional[str]
    runtime: Optional[str]
    port: Union[str, int]  # Can be string (e.g., "8080:8080") or int
    subcommand: Optional[str]


BaseEngineArgs = protocol_to_dataclass(BaseEngineArgsType)


class DefaultArgsType(Protocol):
    container: bool
    runtime: SUPPORTED_RUNTIMES
    store: PathStr
    debug: bool
    quiet: bool
    dryrun: bool
    engine: SUPPORTED_ENGINES
    noout: Optional[bool]


DefaultArgs = protocol_to_dataclass(DefaultArgsType)


class ChatSubArgsType(Protocol):
    prefix: str
    url: str
    color: COLOR_OPTIONS
    list: bool
    model: Optional[str]
    rag: Optional[str]
    api_key: Optional[str]
    ARGS: Optional[List[str]]


ChatSubArgs = protocol_to_dataclass(ChatSubArgsType)


class ChatArgsType(DefaultArgsType, ChatSubArgsType):
    ignore: Optional[bool]  # runtime-only


class ServeRunArgsType(DefaultArgsType, Protocol):
    """Args for serve and run commands"""

    MODEL: str
    port: Optional[int]
    name: Optional[str]
    rag: Optional[str]
    subcommand: str
    detach: Optional[bool]
    api: Optional[str]
    image: str
    host: Optional[str]
    generate: Optional[str]
    context: int
    cache_reuse: int
    authfile: Optional[str]
    device: Optional[list[str]]
    env: list[str]
    ARGS: Optional[list[str]]  # For run command
    mcp: Optional[list[str]]
    summarize_after: int
    # Chat/run specific options
    color: COLOR_OPTIONS
    prefix: str
    rag_image: Optional[str]
    ignore: Optional[bool]  # runtime-only


ServeRunArgs = protocol_to_dataclass(ServeRunArgsType)


class RagArgsType(ServeRunArgsType, Protocol):
    """Args when using RAG functionality - wraps model args"""

    model_args: ServeRunArgsType
    model_host: str
    model_port: int
    rag: str  # type: ignore


RagArgs = protocol_to_dataclass(RagArgsType)
