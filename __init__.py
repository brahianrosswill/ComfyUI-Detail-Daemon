# __init__.py

from .detail_daemon_node import DetailDaemonSamplerNode, DetailDaemonGraphSigmasNode, MultiplySigmas, LyingSigmaSamplerNode, DetailDaemonWan21SamplerNode

NODE_CLASS_MAPPINGS = {
    "DetailDaemonSampler": DetailDaemonSamplerNode,
    "DetailDaemonGraphSigmas": DetailDaemonGraphSigmasNode,
    "MultiplySigmas": MultiplySigmas,
    "LyingSigmaSampler": LyingSigmaSamplerNode,
    "DetailDaemonWan21Sampler": DetailDaemonWan21SamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailDaemonSampler": "Detail Daemon Sampler",
    "DetailDaemonGraphSigmas": "Detail Daemon Graph Sigmas",
    "MultiplySigmas": "Multiply Sigmas",
    "LyingSigmaSampler": "Lying Sigma Sampler",
    "DetailDaemonWan21Sampler": "Detail Daemon Sampler (wan 2.1)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

