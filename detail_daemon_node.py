# Based on the concept from https://github.com/muerrilla/sd-webui-detail-daemon

from __future__ import annotations

import io
import platform
import math
import random
import os

if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from comfy.samplers import KSAMPLER
from PIL import Image
import folder_paths

def make_detail_daemon_schedule(
    steps,
    start,
    end,
    bias,
    amount,
    exponent,
    start_offset,
    end_offset,
    fade,
    smooth,
    artifact_control=0.5,
):
    scaled_amount = math.tanh(amount)
    K = 0.75
    effective_exponent = exponent / (1 + abs(scaled_amount) * K)
    effective_exponent = max(0.1, effective_exponent)

    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    if steps > 0:
        start_idx, mid_idx, end_idx = [
            int(round(x * (steps - 1))) for x in [start, mid, end]
        ]

        # First half of the curve
        num_start = mid_idx - start_idx + 1
        if num_start > 0:
            start_values = np.linspace(0, 1, num_start)
            if smooth:
                start_values = 0.5 * (1 - np.cos(start_values * np.pi))
            start_values = start_values**effective_exponent
            start_values *= (scaled_amount - start_offset)
            start_values += start_offset
            multipliers[start_idx : mid_idx + 1] = start_values

        # Second half of the curve
        num_end = end_idx - mid_idx + 1
        if num_end > 0:
            end_values = np.linspace(1, 0, num_end)
            if smooth:
                end_values = 0.5 * (1 - np.cos(end_values * np.pi))
            end_values = end_values**effective_exponent
            end_values *= (scaled_amount - end_offset)
            end_values += end_offset
            multipliers[mid_idx : end_idx + 1] = end_values

        # Fill offsets
        multipliers[:start_idx] = start_offset
        multipliers[end_idx + 1 :] = end_offset
        
    multipliers *= 1 - fade

    # Artifact control / smoothing
    base_max_delta = 0.15
    min_max_delta = 0.01
    dynamic_max_delta = min_max_delta + (base_max_delta - min_max_delta) * artifact_control

    if steps > 1:
        for i in range(1, steps):
            delta = multipliers[i] - multipliers[i - 1]
            clamped_delta = max(min(delta, dynamic_max_delta), -dynamic_max_delta)
            multipliers[i] = multipliers[i - 1] + clamped_delta

    return multipliers

def get_dd_schedule(
    sigma: float,
    sigmas: torch.Tensor,
    dd_schedule: torch.Tensor,
) -> float:
    sched_len = len(dd_schedule)
    if (
        sched_len < 1
        or len(sigmas) < 2
        or sigma < 0
    ):
        return 0.0
    
    # sigmas is usually descending
    if sigma >= sigmas[0]:
        return dd_schedule[0].item()
    if sigma <= sigmas[-1]:
        return dd_schedule[-1].item()

    # Find the index
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())

    if deltas[idx] == 0:
        return dd_schedule[idx].item()

    # Interpolation
    if sigma > sigmas[idx]:
        idxhigh, idxlow = idx - 1, idx
    else:
        idxhigh, idxlow = idx, idx + 1
    
    if idxhigh < 0:
        return dd_schedule[0].item()
    if idxlow >= sched_len:
        return dd_schedule[sched_len - 1].item()

    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        return dd_schedule[idxlow].item()

    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()

class DetailDaemonGraphSigmasNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "detail_amount": ("FLOAT", {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth": ("BOOLEAN", {"default": True}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "artifact_control": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    OUTPUT_NODE = True
    CATEGORY = "sampling/custom_sampling/sigmas"
    FUNCTION = "make_graph"

    def make_graph(self, sigmas, detail_amount, start, end, bias, exponent, start_offset, end_offset, fade, smooth, cfg_scale, artifact_control):
        out_sigmas = sigmas.clone()
        steps = len(out_sigmas) - 1
        
        schedule = make_detail_daemon_schedule(
            steps, start, end, bias, detail_amount, exponent, start_offset, end_offset, fade, smooth, artifact_control
        )

        for idx in range(steps):
            multiplier = schedule[idx] * 0.1
            out_sigmas[idx] *= (1 - multiplier * cfg_scale)

        image = self.plot_schedule(schedule)
        
        output_dir = folder_paths.get_temp_directory()
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix_append, output_dir)
        filename = f"{filename}_{counter:05}_.png"
        file_path = os.path.join(full_output_folder, filename)
        image.save(file_path, compress_level=1)

        return {
            "ui": {"images": [{"filename": filename, "subfolder": subfolder, "type": "temp"}]},
            "result": (out_sigmas,)
        }

    @staticmethod
    def plot_schedule(schedule) -> Image:
        plt.figure(figsize=(6, 4))
        plt.plot(schedule, label="Sigma Adjustment Curve")
        plt.xlabel("Steps")
        plt.ylabel("Multiplier (*10)")
        plt.title("Detail Adjustment Schedule")
        plt.legend()
        plt.grid(True)
        if len(schedule) > 0:
            plt.xticks(np.linspace(0, len(schedule)-1, min(len(schedule), 11)).astype(int))
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG")
        plt.close()
        buf.seek(0)
        image = Image.open(buf)
        return image

def detail_daemon_sampler(model, x, sigmas, *, dds_wrapped_sampler, dds_make_schedule, dds_cfg_scale_override, **kwargs):
    if dds_cfg_scale_override > 0:
        cfg_scale = dds_cfg_scale_override
    else:
        # Improved CFG extraction
        cfg_scale = 1.0
        if hasattr(model, "inner_model"):
            cfg_scale = getattr(model.inner_model, "cfg", 1.0)
        elif hasattr(model, "cfg"):
            cfg_scale = getattr(model, "cfg", 1.0)
            
    if not isinstance(cfg_scale, (int, float)):
        cfg_scale = 1.0

    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1])

    def model_wrapper(x, sigma, **extra_args):
        # sigma is a tensor, often with 1 element or matching batch size
        sigma_float = float(sigma.flatten()[0].detach().cpu())
        
        # Only adjust if within range
        if sigma_min <= sigma_float <= sigma_max:
            dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
            # Apply adjustment
            adjusted_sigma = sigma * max(0.001, 1.0 - dd_adjustment * cfg_scale)
            return model(x, adjusted_sigma, **extra_args)
        
        return model(x, sigma, **extra_args)

    # Patch attributes so the sampler can find what it needs
    for k in ("inner_model", "sigmas", "model_sampling"):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))

    return dds_wrapped_sampler.sampler_function(
        model_wrapper, x, sigmas, **kwargs, **dds_wrapped_sampler.extra_options
    )

class DetailDaemonSamplerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "detail_amount": ("FLOAT", {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth": ("BOOLEAN", {"default": True}),
                "cfg_scale_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "artifact_control": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "go"

    def go(self, sampler, **kwargs):
        def dds_make_schedule(steps):
            return make_detail_daemon_schedule(
                steps,
                kwargs['start'], kwargs['end'], kwargs['bias'], kwargs['detail_amount'],
                kwargs['exponent'], kwargs['start_offset'], kwargs['end_offset'],
                kwargs['fade'], kwargs['smooth'], kwargs['artifact_control']
            )

        return (
            KSAMPLER(
                detail_daemon_sampler,
                extra_options={
                    "dds_wrapped_sampler": sampler,
                    "dds_make_schedule": dds_make_schedule,
                    "dds_cfg_scale_override": kwargs['cfg_scale_override'],
                },
            ),
        )

# Wan 2.1 Video Generator Version
def detail_daemon_wan21_sampler(model, x, sigmas, *, dds_wrapped_sampler, dds_make_schedule, **kwargs):
    # Wan 2.1 often uses its own way of handling CFG
    cfg_scale = 1.0
    if hasattr(model, "inner_model"):
        cfg_scale = getattr(model.inner_model, "cfg", 1.0)
    elif hasattr(model, "cfg"):
        cfg_scale = getattr(model, "cfg", 1.0)
    
    if not isinstance(cfg_scale, (int, float)):
        cfg_scale = 1.0

    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1])

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.flatten()[0].detach().cpu())
        if sigma_min <= sigma_float <= sigma_max:
            dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
            adjusted_sigma = sigma * max(0.001, 1.0 - dd_adjustment * cfg_scale)
            return model(x, adjusted_sigma, **extra_args)
        return model(x, sigma, **extra_args)

    for k in ("inner_model", "sigmas", "model_sampling"):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))

    return dds_wrapped_sampler.sampler_function(
        model_wrapper, x, sigmas, **kwargs, **dds_wrapped_sampler.extra_options
    )

class DetailDaemonWan21SamplerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "detail_amount": ("FLOAT", {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth": ("BOOLEAN", {"default": True}),
                "artifact_control": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers/wan2.1"
    FUNCTION = "go"

    def go(self, sampler, **kwargs):
        def dds_make_schedule(steps):
            return make_detail_daemon_schedule(
                steps,
                kwargs['start'], kwargs['end'], kwargs['bias'], kwargs['detail_amount'],
                kwargs['exponent'], kwargs['start_offset'], kwargs['end_offset'],
                kwargs['fade'], kwargs['smooth'], kwargs['artifact_control']
            )

        return (
            KSAMPLER(
                detail_daemon_wan21_sampler,
                extra_options={
                    "dds_wrapped_sampler": sampler,
                    "dds_make_schedule": dds_make_schedule,
                },
            ),
        )

class MultiplySigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    FUNCTION = "multiply"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def multiply(self, sigmas, factor, start_percent, end):
        # Clone the sigmas to ensure the input is not modified (stateless)
        sigmas = sigmas.clone()
        
        total_sigmas = len(sigmas)
        start_idx = int(start * total_sigmas)
        end_idx = int(end * total_sigmas)

        for i in range(start_idx, end_idx):
            sigmas[i] *= factor

        return (sigmas,)

#LyingSigmaSampler
def lying_sigma_sampler(
    model,
    x,
    sigmas,
    *,
    lss_wrapped_sampler,
    lss_dishonesty_factor,
    lss_startend_percent,
    **kwargs,
):
    start_percent, end_percent = lss_startend_percent
    ms = model.inner_model.inner_model.model_sampling
    start_sigma, end_sigma = (
        round(ms.percent_to_sigma(start_percent), 4),
        round(ms.percent_to_sigma(end_percent), 4),
    )
    del ms

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        if end_sigma <= sigma_float <= start_sigma:
            sigma = sigma * (1.0 + lss_dishonesty_factor)
        return model(x, sigma, **extra_args)

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return lss_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **lss_wrapped_sampler.extra_options,
    )


class LyingSigmaSamplerNode:
    CATEGORY = "sampling/custom_sampling"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "dishonesty_factor": (
                    "FLOAT",
                    {
                        "default": -0.05,
                        "min": -0.999,
                        "step": 0.01,
                        "tooltip": "Multiplier for sigmas passed to the model. -0.05 means we reduce the sigma by 5%.",
                    },
                ),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    @classmethod
    def go(cls, sampler, dishonesty_factor, *, start_percent=0.0, end_percent=1.0):
        return (
            KSAMPLER(
                lying_sigma_sampler,
                extra_options={
                    "lss_wrapped_sampler": sampler,
                    "lss_dishonesty_factor": dishonesty_factor,
                    "lss_startend_percent": (start_percent, end_percent),
                },
            ),
        )

