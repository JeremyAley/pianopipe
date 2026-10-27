"""Quest Piano Trainer content pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pianopipe")  # pyproject.toml is the one place the version is written
except PackageNotFoundError:  # run from a source tree without installing
    __version__ = "0+unknown"
