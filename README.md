# ComfyUI-Detail-Daemon
A port of muerrilla's sd-webui-Detail-Daemon as a node for ComfyUI, to adjust sigmas that control detail.

## Nodes

### Detail Daemon Sampler

Allows sampling with the Detail Daemon schedule adjustment.

### Detail Daemon Graph Sigmas

Allows graphing adjusted sigmas to visually see the effects of different parameters.

### Multiply Sigmas

Simple node to multiply sigmas by the supplied factor.

## Credits

Concept and schedule generation function from: https://github.com/muerrilla/sd-webui-detail-daemon/

ComfyUI sampler implementation and schedule interpolation by https://github.com/blepping/
