"""Compatibility wrapper for `python -m deployment.demo.health`."""

from .cli.health import main


if __name__ == "__main__":
    main()
