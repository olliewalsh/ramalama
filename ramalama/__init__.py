"""ramalama client module."""

import sys

from ramalama.cli import HelpException, init_cli, print_version
from ramalama.common import perror

assert sys.version_info >= (3, 9), "Python 3.9 or greater is required."

__all__ = ["perror", "init_cli", "print_version", "HelpException"]
