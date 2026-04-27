import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

_module_logger = logging.getLogger(__name__)


# PI v2.5.0 environment variable keys.
ENV_INTERFACE_VERSION = "PI_INTERFACE_VERSION"
ENV_CLIENT_NAME = "PI_CLIENT_NAME"
ENV_CLIENT_VERSION = "PI_CLIENT_VERSION"
ENV_CLIENT_LANGUAGE = "PI_CLIENT_LANGUAGE"
ENV_CLIENT_MAAFW_VERSION = "PI_CLIENT_MAAFW_VERSION"
ENV_VERSION = "PI_VERSION"
ENV_CONTROLLER = "PI_CONTROLLER"
ENV_RESOURCE = "PI_RESOURCE"


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_as_string(item) for item in value if item is not None]


@dataclass(frozen=True)
class Win32Config:
    class_regex: str = ""
    window_regex: str = ""
    screencap: str = ""
    mouse: str = ""
    keyboard: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "Win32Config | None":
        if not isinstance(data, dict):
            return None
        return cls(
            class_regex=_as_string(data.get("class_regex")),
            window_regex=_as_string(data.get("window_regex")),
            screencap=_as_string(data.get("screencap")),
            mouse=_as_string(data.get("mouse")),
            keyboard=_as_string(data.get("keyboard")),
        )


@dataclass(frozen=True)
class MacOSConfig:
    title_regex: str = ""
    screencap: str = ""
    input: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "MacOSConfig | None":
        if not isinstance(data, dict):
            return None
        return cls(
            title_regex=_as_string(data.get("title_regex")),
            screencap=_as_string(data.get("screencap")),
            input=_as_string(data.get("input")),
        )


@dataclass(frozen=True)
class PlayCoverConfig:
    uuid: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "PlayCoverConfig | None":
        if not isinstance(data, dict):
            return None
        return cls(uuid=_as_string(data.get("uuid")))


@dataclass(frozen=True)
class GamepadConfig:
    class_regex: str = ""
    window_regex: str = ""
    gamepad_type: str = ""
    screencap: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "GamepadConfig | None":
        if not isinstance(data, dict):
            return None
        return cls(
            class_regex=_as_string(data.get("class_regex")),
            window_regex=_as_string(data.get("window_regex")),
            gamepad_type=_as_string(data.get("gamepad_type")),
            screencap=_as_string(data.get("screencap")),
        )


@dataclass(frozen=True)
class Controller:
    name: str = ""
    label: str = ""
    description: str = ""
    icon: str = ""
    type: str = ""
    display_short_side: int | None = None
    display_long_side: int | None = None
    display_raw: bool | None = None
    permission_required: bool = False
    attach_resource_path: list[str] = field(default_factory=list)
    option: list[str] = field(default_factory=list)
    win32: Win32Config | None = None
    adb: Any = None
    macos: MacOSConfig | None = None
    playcover: PlayCoverConfig | None = None
    gamepad: GamepadConfig | None = None
    wlroots: Any = None

    @classmethod
    def from_dict(cls, data: Any) -> "Controller":
        if not isinstance(data, dict):
            raise TypeError("PI_CONTROLLER is not a JSON object")
        return cls(
            name=_as_string(data.get("name")),
            label=_as_string(data.get("label")),
            description=_as_string(data.get("description")),
            icon=_as_string(data.get("icon")),
            type=_as_string(data.get("type")),
            display_short_side=_as_int(data.get("display_short_side")),
            display_long_side=_as_int(data.get("display_long_side")),
            display_raw=_as_bool(data.get("display_raw")),
            permission_required=bool(data.get("permission_required", False)),
            attach_resource_path=_as_string_list(data.get("attach_resource_path")),
            option=_as_string_list(data.get("option")),
            win32=Win32Config.from_dict(data.get("win32")),
            adb=data.get("adb"),
            macos=MacOSConfig.from_dict(data.get("macos")),
            playcover=PlayCoverConfig.from_dict(data.get("playcover")),
            gamepad=GamepadConfig.from_dict(data.get("gamepad")),
            wlroots=data.get("wlroots"),
        )


@dataclass(frozen=True)
class Resource:
    name: str = ""
    label: str = ""
    description: str = ""
    icon: str = ""
    path: list[str] = field(default_factory=list)
    controller: list[str] = field(default_factory=list)
    option: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "Resource":
        if not isinstance(data, dict):
            raise TypeError("PI_RESOURCE is not a JSON object")
        return cls(
            name=_as_string(data.get("name")),
            label=_as_string(data.get("label")),
            description=_as_string(data.get("description")),
            icon=_as_string(data.get("icon")),
            path=_as_string_list(data.get("path")),
            controller=_as_string_list(data.get("controller")),
            option=_as_string_list(data.get("option")),
        )


@dataclass(frozen=True)
class Env:
    interface_version: str = ""
    client_name: str = ""
    client_version: str = ""
    client_language: str = ""
    client_maafw_version: str = ""
    version: str = ""
    controller: Controller | None = None
    controller_raw: str = ""
    resource: Resource | None = None
    resource_raw: str = ""


_global_env: Env | None = None
_init_lock = threading.Lock()


def _parse_json_env(env_key: str, raw: str, parser: Any) -> Any:
    if not raw:
        return None

    try:
        return parser(json.loads(raw))
    except Exception as exc:
        _module_logger.warning("failed to parse %s: %s", env_key, exc)
        return None


def _build_env() -> Env:
    controller_raw = os.getenv(ENV_CONTROLLER, "")
    resource_raw = os.getenv(ENV_RESOURCE, "")

    env = Env(
        interface_version=os.getenv(ENV_INTERFACE_VERSION, ""),
        client_name=os.getenv(ENV_CLIENT_NAME, ""),
        client_version=os.getenv(ENV_CLIENT_VERSION, ""),
        client_language=os.getenv(ENV_CLIENT_LANGUAGE, ""),
        client_maafw_version=os.getenv(ENV_CLIENT_MAAFW_VERSION, ""),
        version=os.getenv(ENV_VERSION, ""),
        controller_raw=controller_raw,
        resource_raw=resource_raw,
        controller=_parse_json_env(
            ENV_CONTROLLER, controller_raw, Controller.from_dict
        ),
        resource=_parse_json_env(ENV_RESOURCE, resource_raw, Resource.from_dict),
    )

    _module_logger.info(
        "PI environment initialized: interface_version=%s client_name=%s client_version=%s client_language=%s client_maafw_version=%s pi_version=%s controller_ok=%s resource_ok=%s",
        env.interface_version,
        env.client_name,
        env.client_version,
        env.client_language,
        env.client_maafw_version,
        env.version,
        env.controller is not None,
        env.resource is not None,
    )
    return env


def init(force: bool = False) -> Env:
    global _global_env
    with _init_lock:
        if force or _global_env is None:
            _global_env = _build_env()
    return _global_env


def get() -> Env:
    return init()


def interface_version() -> str:
    return get().interface_version


def client_name() -> str:
    return get().client_name


def client_version() -> str:
    return get().client_version


def client_language() -> str:
    return get().client_language


def client_maafw_version() -> str:
    return get().client_maafw_version


def project_version() -> str:
    return get().version


def controller() -> Controller | None:
    return get().controller


def resource() -> Resource | None:
    return get().resource


def controller_type() -> str:
    current = controller()
    return current.type if current else ""


def controller_name() -> str:
    current = controller()
    return current.name if current else ""


def resource_name() -> str:
    current = resource()
    return current.name if current else ""


def resource_label() -> str:
    current = resource()
    if not current:
        return ""
    return current.label or current.name


def resource_paths() -> list[str]:
    current = resource()
    if not current:
        return []
    return list(current.path)


__all__ = [
    "ENV_INTERFACE_VERSION",
    "ENV_CLIENT_NAME",
    "ENV_CLIENT_VERSION",
    "ENV_CLIENT_LANGUAGE",
    "ENV_CLIENT_MAAFW_VERSION",
    "ENV_VERSION",
    "ENV_CONTROLLER",
    "ENV_RESOURCE",
    "Win32Config",
    "MacOSConfig",
    "PlayCoverConfig",
    "GamepadConfig",
    "Controller",
    "Resource",
    "Env",
    "init",
    "get",
    "interface_version",
    "client_name",
    "client_version",
    "client_language",
    "client_maafw_version",
    "project_version",
    "controller",
    "resource",
    "controller_type",
    "controller_name",
    "resource_name",
    "resource_label",
    "resource_paths",
]