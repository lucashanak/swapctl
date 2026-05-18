"""swapctl backend package."""


class SwapctlError(Exception):
    """Base exception for swapctl."""


class ConfigError(SwapctlError):
    """Raised on config.yaml read/write/validation failure."""


class DownloadError(SwapctlError):
    """Raised on download/install failure."""


__all__ = ["SwapctlError", "ConfigError", "DownloadError"]
