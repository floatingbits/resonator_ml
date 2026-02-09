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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", type=str)
    train.add_argument("--set", action="append", default=[])

    out = subparsers.add_parser("out")
    out.add_argument("--config", type=str)
    out.add_argument("--set", action="append", default=[])

    return parser
