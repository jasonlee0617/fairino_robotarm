def param(node, name: str, default):
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


def param_f(node, name: str, default: float) -> float:
    return float(param(node, name, default))


def param_b(node, name: str, default: bool) -> bool:
    return bool(param(node, name, default))


__all__ = ["param", "param_b", "param_f"]
