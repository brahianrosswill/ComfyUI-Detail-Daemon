# ComfyUI-Detail-Daemon
A port of muerrilla's sd-webui-Detail-Daemon as a node for ComfyUI, to adjust sigmas that enhance detail, and possibly remove unwanted bokeh or background blurring, particularly with Flux models.

## Nodes

### Detail Daemon Sampler

Allows sampling with the Detail Daemon schedule adjustment, which keeps the noise levels injected the same while lowering the amount of noise removed at each step, which effectively adds detail. Detail_amounts between 0 and 1.0 work best. See muerrilla's [Detail Daemon](https://github.com/muerrilla/sd-webui-detail-daemon/) repo for full explanation of inputs and methodology.

### Detail Daemon Graph Sigmas

Allows graphing adjusted sigmas to visually see the effects of different parameters. This had to be a separate node from the above node in order to function properly. Just set the values the same.

### Multiply Sigmas

Simple node to multiply all sigmas by the supplied factor (multiplies both the noise levels added and denoised by the factor, which somehow adds detail with a factor less than 1). Values of 0.95-0.99 work best (default without this node is 1.0). It is stateless, meaning it calculates the sigmas fresh on every queue (other Multiply Sigmas nodes seem to calculate on prior run sigmas). Because this multiplies sigmas of all steps (without start or end values), it tends to change the overall composition of the image too.

### Lying Sigma Sampler

A simpler version of Detail Daemon Sampler, with only amount adjustment (-0.05 dishonesty_factor is equivalent of 0.5 in detail_amount of Detail Daemon), start and end values. Dishonesty values between -0.1 and -0.01 work best.

## Credits

Detail Daemon concept and schedule generation function from: https://github.com/muerrilla/sd-webui-detail-daemon/

ComfyUI sampler implementation and schedule interpolation, as well as Lying Sigma Sampler, by https://github.com/blepping/
