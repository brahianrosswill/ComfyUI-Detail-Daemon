# __init__.py

from .detail_daemon_node import DetailDaemonNode, MultiplySigmas

NODE_CLASS_MAPPINGS = {
    "DetailDaemonNode": DetailDaemonNode,
    "MultiplySigmas": MultiplySigmas
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailDaemonNode": "Detail Daemon",
    "MultiplySigmas": "Multiply Sigmas (stateless)"
}

__all__ = ["NODE_CLASS_MAPPINGS"]

