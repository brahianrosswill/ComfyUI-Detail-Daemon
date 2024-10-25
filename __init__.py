# __init__.py

from .detail_daemon_node import DetailDaemonSamplerNode, DetailDaemonGraphSigmasNode, MultiplySigmas

NODE_CLASS_MAPPINGS = {
    "DetailDaemonSamplerNode": DetailDaemonSamplerNode,
    "DetailDaemonGraphSigmasNode": DetailDaemonGraphSigmasNode,
    "MultiplySigmas": MultiplySigmas
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailDaemonSamplerNode": "Detail Daemon Sampler",
    "DetailDaemonGraphSigmasNode": "Detail Daemon Graph Sigmas",
    "MultiplySigmas": "Multiply Sigmas (stateless)"
}

__all__ = ["NODE_CLASS_MAPPINGS"]

