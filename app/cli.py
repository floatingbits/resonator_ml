import argparse
from typing import Dict, Any


def parse_set_arguments(set_args: list[str]) -> Dict[str, Any]:
    overrides = {}

    for item in set_args:
        key, value = item.split("=", 1)

        # primitive Typ-Inferenz
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        current = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    return overrides


def build_parser(command_names: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)


    for command_name in command_names:
        subparser = subparsers.add_parser(command_name)
        subparser.add_argument("--config", type=str)
        subparser.add_argument("--set", action="append", default=[])

    return parser
