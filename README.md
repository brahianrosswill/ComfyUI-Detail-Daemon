![DetailDaemon-example](https://github.com/user-attachments/assets/8f336c94-a4c6-426e-abe1-6a4c80a37cbb)
# ComfyUI-Detail-Daemon
A port of muerrilla's sd-webui-Detail-Daemon as a node for ComfyUI, to adjust sigmas that generally enhance details, and possibly remove unwanted bokeh or background blurring, particularly with Flux models. If the values are taken too far it results in an oversharpened and/or HDR effect.

## Nodes

### Detail Daemon Sampler

![Screenshot 2024-10-29 124741](https://github.com/user-attachments/assets/c11bd716-1fa1-43b6-8d64-ab20642bceb5)

Allows sampling with the Detail Daemon schedule adjustment, which keeps the noise levels injected the same while lowering the amount of noise removed at each step, which effectively adds detail. Detail_amounts between 0 and 1.0 work best. See muerrilla's [Detail Daemon](https://github.com/muerrilla/sd-webui-detail-daemon/) repo for full explanation of inputs and methodology.

### Detail Daemon Graph Sigmas

![Screenshot 2024-10-29 131939](https://github.com/user-attachments/assets/d0a3f895-5f6d-4b94-b4d1-aa86e7acb5d7)

Allows graphing adjusted sigmas to visually see the effects of different parameters. This had to be a separate node from the Detail Daemon Sampler node in order to function properly. Just set the values the same, or set inputs on separate nodes that feed into both the Detail Daemon Sampler and this Graph Sigmas node. You'll need to run the queue in order to see the graph on the node.

### Multiply Sigmas

![Screenshot 2024-10-29 124833](https://github.com/user-attachments/assets/25efbad7-8df2-4c21-a7b5-989d2954df48)

Simple node to multiply all sigmas by the supplied factor (multiplies both the noise levels added and denoised by the factor, which somehow adds detail with a factor less than 1). Factor values of 0.95-0.99 work best (default without this node is 1.0). It is stateless, meaning it calculates the sigmas fresh on every queue (other Multiply Sigmas nodes seem to calculate on prior run sigmas). Because this multiplies sigmas of all steps (without start or end values), it tends to change the overall composition of the image too.

### Lying Sigma Sampler

![Screenshot 2024-10-29 124803](https://github.com/user-attachments/assets/11c24b49-96e1-4f50-9b82-1d6778c2a8ea)

A simpler version of Detail Daemon Sampler, with only amount adjustment (-0.05 dishonesty_factor is equivalent of 0.5 in detail_amount of Detail Daemon), start and end values. Dishonesty values between -0.1 and -0.01 work best.

## Example and testing workflow

![Screenshot 2024-10-29 125148](https://github.com/user-attachments/assets/ca600acf-5d21-42a5-81d5-b5970f1384cb)

The `Comparing Detailers.json` workflow will allow you to compare these various detailer nodes on the same prompt and seed.

## Credits

Detail Daemon concept and schedule generation function from: https://github.com/muerrilla/sd-webui-detail-daemon/

ComfyUI sampler implementation and schedule interpolation, as well as Lying Sigma Sampler, by https://github.com/blepping/
