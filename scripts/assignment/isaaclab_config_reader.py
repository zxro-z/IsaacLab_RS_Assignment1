"""Read Isaac Lab YAML as inert, JSON-compatible syntax data, never objects.

Python tags (including builtins.slice) are preserved as tag/value envelopes.
No Python constructor, import, callable or UnsafeLoader is ever invoked. Raw
YAML remains untouched; its bytes/hash remain the provenance source of truth.
"""

import math
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

STANDARD = "tag:yaml.org,2002:"
SCALARS = {STANDARD + name for name in ("str", "null", "bool", "int", "float")}


def parse_config_text(text):
    if len(text.encode("utf-8")) > 4 * 1024 * 1024:
        raise ValueError("Config exceeds the 4 MiB reader limit")
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    scalar_loader = yaml.SafeLoader("")
    active, count = set(), 0

    def convert(node, depth=0):
        nonlocal count
        count += 1
        if depth > 100 or count > 100000 or id(node) in active:
            raise ValueError("Cyclic/oversized YAML is not a supported config")
        active.add(id(node))
        try:
            if isinstance(node, ScalarNode):
                if node.tag in SCALARS:
                    value = scalar_loader.construct_object(node)
                    if isinstance(value, float) and not math.isfinite(value):
                        return {"__yaml_tag__": node.tag, "value": node.value}
                    return value
                value = node.value
            elif isinstance(node, SequenceNode):
                value = [convert(item, depth + 1) for item in node.value]
                if node.tag == STANDARD + "seq":
                    return value
            elif isinstance(node, MappingNode):
                value = {}
                for key_node, item in node.value:
                    key = convert(key_node, depth + 1)
                    if not isinstance(key, str) or key in value:
                        raise ValueError("Config keys must be unique strings")
                    value[key] = convert(item, depth + 1)
                if node.tag == STANDARD + "map":
                    return value
            else:
                raise ValueError("Unsupported YAML node")
            return {"__yaml_tag__": node.tag, "value": value}
        finally:
            active.remove(id(node))

    try:
        return convert(root) if root is not None else None
    finally:
        scalar_loader.dispose()


def load_config(path):
    return parse_config_text(Path(path).read_text())
