from dataclasses import is_dataclass, fields


def merge_dataclass(instance, overrides: dict):
    for f in fields(instance):
        if f.name not in overrides:
            continue

        value = getattr(instance, f.name)
        override = overrides[f.name]

        if is_dataclass(value) and isinstance(override, dict):
            merge_dataclass(value, override)
        else:
            setattr(instance, f.name, override)
