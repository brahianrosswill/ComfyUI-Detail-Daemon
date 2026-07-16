# Based on the concept from https://github.com/muerrilla/sd-webui-detail-daemon

from __future__ import annotations

import io
import math

# Trying matplotlib NSWindow warning workaround on macOS
import platform

if platform.system() == 'Darwin':  # Check if running on macOS
    import matplotlib
    matplotlib.use('Agg')  # Set non-GUI backend to avoid crashes

import matplotlib.pyplot as plt
import numpy as np
import torch
from comfy.samplers import KSAMPLER
from PIL import Image
import folder_paths
import random
import os


# Schedule creation function from https://github.com/muerrilla/sd-webui-detail-daemon
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
):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    return multipliers


class DetailDaemonGraphSigmasNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "detail_amount": (
                    "FLOAT",
                    {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "start": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "bias": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "exponent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
                "start_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "end_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "fade": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "smooth": ("BOOLEAN", {"default": True}),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "round": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "sampling/custom_sampling/sigmas"
    FUNCTION = "make_graph"

    def make_graph(
        self,
        sigmas,
        detail_amount,
        start,
        end,
        bias,
        exponent,
        start_offset,
        end_offset,
        fade,
        smooth,
        cfg_scale,
    ):
        # Create a copy of the input sigmas using clone() for tensors to avoid modifying the original
        sigmas = sigmas.clone()

        # Derive the number of steps from the length of sigmas minus 1 (ignore the final sigma)
        steps = len(sigmas) - 1  # 21 sigmas, 20 steps
        actual_steps = steps

        # Create the schedule using the number of steps
        schedule = make_detail_daemon_schedule(
            actual_steps,
            start,
            end,
            bias,
            detail_amount,
            exponent,
            start_offset,
            end_offset,
            fade,
            smooth,
        )

        # Debugging: print schedule and sigmas lengths to verify alignment
        print(
            f"Number of sigmas: {len(sigmas)}, Number of schedule steps: {len(schedule)}",
        )

        # Iterate over the sigmas, except for the last one (which we assume is 0 and leave untouched)
        for idx in range(steps):
            multiplier = schedule[idx] * 0.1

            # Debugging: print each index and sigma to track what's being adjusted
            print(f"Adjusting sigma at index {idx} with multiplier {multiplier}")

            sigmas[idx] *= (
                1 - multiplier * cfg_scale
            )  # Adjust each sigma in "both" mode

        # Create the plot for visualization
        image = self.plot_schedule(schedule)
        
        # Save temp image
        output_dir = folder_paths.get_temp_directory()
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        
        full_output_folder, filename, counter, subfolder, _ = (
        folder_paths.get_save_image_path(prefix_append, output_dir)
        )
        filename = f"{filename}_{counter:05}_.png"
        file_path = os.path.join(full_output_folder, filename)
        image.save(file_path, compress_level=1)

        return {
            "ui": {
                "images": [
                    {"filename": filename, "subfolder": subfolder, "type": "temp"},
                ],
            }
        }


    @staticmethod
    def plot_schedule(schedule) -> Image:
        plt.figure(figsize=(6, 4))  # Adjusted width
        plt.plot(schedule, label="Sigma Adjustment Curve")
        plt.xlabel("Steps")
        plt.ylabel("Multiplier (*10)")
        plt.title("Detail Adjustment Schedule")
        plt.legend()
        plt.grid(True)
        plt.xticks(range(len(schedule)))
        plt.ylim(-1, 1)

        # Use tight_layout or subplots_adjust
        plt.tight_layout()
        # Or manually adjust if needed:
        # plt.subplots_adjust(left=0.2)

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG")
        plt.close()
        buf.seek(0)
        image = Image.open(buf)
        return image


def get_dd_schedule(
    sigma: float,
    sigmas: torch.Tensor,
    dd_schedule: torch.Tensor,
) -> float:
    sched_len = len(dd_schedule)
    if (
        sched_len < 2
        or len(sigmas) < 2
        or sigma <= 0
        or not (sigmas[-1] <= sigma <= sigmas[0])
    ):
        return 0.0
    # First, we find the index of the closest sigma in the list to what the model was
    # called with.
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())
    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2])
        or deltas[idx] == 0
    ):
        # Either exact match or closest to head/tail of the DD schedule so we
        # can't interpolate to another schedule item.
        return dd_schedule[idx].item()
    # If we're here, that means the sigma is in between two sigmas in the
    # list.
    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    # We find the low/high neighbor sigmas - our sigma is somewhere between them.
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        # Shouldn't be possible, but just in case... Avoid divide by zero.
        return dd_schedule[idxlow]
    # Ratio of how close we are to the high neighbor.
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    # Mix the DD schedule high/low items according to the ratio.
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()


def detail_daemon_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    dds_wrapped_sampler: object,
    dds_make_schedule: callable,
    dds_cfg_scale_override: float,
    **kwargs: dict,
) -> torch.Tensor:
    if dds_cfg_scale_override > 0:
        cfg_scale = dds_cfg_scale_override
    else:
        maybe_cfg_scale = getattr(model.inner_model, "cfg", None)
        cfg_scale = (
            float(maybe_cfg_scale) if isinstance(maybe_cfg_scale, (int, float)) else 1.0
        )
    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict):
        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_min <= sigma_float <= sigma_max):
            return model(x, sigma, **extra_args)
        dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)
        return model(x, adjusted_sigma, **extra_args)

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return dds_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **dds_wrapped_sampler.extra_options,
    )


class DetailDaemonSamplerNode:
    DESCRIPTION = "This sampler wrapper works by adjusting the sigma passed to the model, while the rest of sampling stays the same."
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "detail_amount": (
                    "FLOAT",
                    {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "start": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "bias": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "exponent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
                "start_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "end_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "fade": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "smooth": ("BOOLEAN", {"default": True}),
                "cfg_scale_override": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "round": 0.01,
                        "tooltip": "If set to 0, the sampler will automatically determine the CFG scale (if possible). Set to some other value to override.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        sampler: object,
        *,
        detail_amount,
        start,
        end,
        bias,
        exponent,
        start_offset,
        end_offset,
        fade,
        smooth,
        cfg_scale_override,
    ) -> tuple:
        def dds_make_schedule(steps):
            return make_detail_daemon_schedule(
                steps,
                start,
                end,
                bias,
                detail_amount,
                exponent,
                start_offset,
                end_offset,
                fade,
                smooth,
            )

        return (
            KSAMPLER(
                detail_daemon_sampler,
                extra_options={
                    "dds_wrapped_sampler": sampler,
                    "dds_make_schedule": dds_make_schedule,
                    "dds_cfg_scale_override": cfg_scale_override,
                },
            ),
        )

#MultiplySigmas Node
class MultiplySigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "factor": ("FLOAT", {"default": 1, "min": 0, "max": 100, "step": 0.001}),
                "start": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.001}),
                "end": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.001})
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas, factor, start, end):
        # Clone the sigmas to ensure the input is not modified (stateless)
        sigmas = sigmas.clone()
        
        total_sigmas = len(sigmas)
        start_idx = int(start * total_sigmas)
        end_idx = int(end * total_sigmas)

        for i in range(start_idx, end_idx):
            sigmas[i] *= factor

        return (sigmas,)

#LyingSigmaSampler
def _lss_fade_weight(t: float, fade: float) -> float:
    """Cosine ease-in/out weight for a normalized position t in [0, 1].

    The weight is 0 at both range boundaries, ramps up over the first `fade`
    fraction of the range, stays at 1 in the middle, and ramps back down over
    the last `fade` fraction. With fade=0 the original hard on/off behavior is
    preserved (weight is always 1 inside the range).
    """
    if fade <= 0.0:
        return 1.0
    fade = min(fade, 0.5)
    if t <= 0.0 or t >= 1.0:
        return 0.0
    if t < fade:
        return 0.5 * (1.0 - math.cos(math.pi * t / fade))
    if t > 1.0 - fade:
        return 0.5 * (1.0 - math.cos(math.pi * (1.0 - t) / fade))
    return 1.0


def _lss_step_distance(
    sigmas_cpu: torch.Tensor, sigma_float: float, direction: float
) -> float:
    """Distance from sigma_float to the adjacent scheduled sigma in the
    direction of the lie (direction < 0: next/lower sigma, > 0: previous/higher
    sigma). Returns 0 when there is no neighbor in that direction."""
    if direction < 0:
        neighbors = sigmas_cpu[sigmas_cpu < sigma_float - 1e-08]
        if neighbors.numel() == 0:
            return 0.0
        return sigma_float - float(neighbors.max())
    neighbors = sigmas_cpu[sigmas_cpu > sigma_float + 1e-08]
    if neighbors.numel() == 0:
        return 0.0
    return float(neighbors.min()) - sigma_float


def lying_sigma_sampler(
    model,
    x,
    sigmas,
    *,
    lss_wrapped_sampler,
    lss_dishonesty_factor,
    lss_startend_percent,
    lss_fade=0.0,
    lss_max_step_overshoot=0.0,
    **kwargs,
):
    start_percent, end_percent = lss_startend_percent
    # Guard against swapped bounds, which would silently disable the effect.
    if start_percent > end_percent:
        start_percent, end_percent = end_percent, start_percent
    ms = model.inner_model.inner_model.model_sampling
    start_sigma, end_sigma = (
        round(ms.percent_to_sigma(start_percent), 4),
        round(ms.percent_to_sigma(end_percent), 4),
    )
    del ms

    sigmas_cpu = sigmas.detach().clone().cpu()
    # Normalized positions for the fade are measured in log-sigma space so the
    # ramp spreads evenly across steps (real schedules are roughly log-spaced
    # and much denser at low sigmas). end_sigma can be 0, so floor it for log.
    log_start = math.log(max(start_sigma, 1e-06))
    log_end = math.log(max(end_sigma, 1e-06))
    log_span = max(log_start - log_end, 1e-08)

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        if sigma_float <= 0.0 or not (end_sigma <= sigma_float <= start_sigma):
            return model(x, sigma, **extra_args)
        factor = lss_dishonesty_factor
        if lss_fade > 0.0:
            # Normalized position within the active range
            # (0 = end/low-sigma boundary, 1 = start/high-sigma boundary).
            t = (math.log(sigma_float) - log_end) / log_span
            factor *= _lss_fade_weight(t, lss_fade)
        if factor != 0.0:
            adjusted = sigma_float * (1.0 + factor)
            if lss_max_step_overshoot > 0.0:
                # Limit the lie to lss_max_step_overshoot times the local step
                # size. Prevents the model from being told the latent is
                # cleaner/dirtier than the schedule can actually deliver,
                # which is the main cause of fried/oversharpened artifacts.
                step_dist = _lss_step_distance(sigmas_cpu, sigma_float, factor)
                if step_dist > 0.0:
                    max_lie = step_dist * lss_max_step_overshoot
                    if factor < 0:
                        adjusted = max(adjusted, sigma_float - max_lie)
                    else:
                        adjusted = min(adjusted, sigma_float + max_lie)
            # Never pass a zero/negative sigma to the model.
            adjusted = max(adjusted, 1e-06)
            sigma = sigma * (adjusted / sigma_float)
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
    DESCRIPTION = "Lies about the current sigma to increase detail (negative factor) or smooth/reduce it (positive factor). Includes fade and overshoot-guard options to reduce artifacts at stronger settings."
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
                        "tooltip": "Multiplier for sigmas passed to the model. -0.05 means we reduce the sigma by 5%. Negative values increase detail; combine with fade/max_step_overshoot to avoid artifacts at stronger values.",
                    },
                ),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fade": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Smoothly eases the effect in and out over this fraction of the active range (cosine ramp, spread evenly across steps), instead of switching on/off abruptly at the start/end boundaries. Reduces banding/transition artifacts. 0 = disabled (original behavior), 1 = gentle ramp over the whole range.",
                    },
                ),
                "max_step_overshoot": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Limits how far the lie may push the effective sigma, in units of the current step size. 1.0 = never lie past the next scheduled sigma, preventing over-denoising (fried/oversharpened/HDR artifacts). 0 = no limit (original behavior).",
                    },
                ),
            },
        }

    @classmethod
    def go(cls, sampler, dishonesty_factor, *, start_percent=0.1, end_percent=0.9, fade=0.0, max_step_overshoot=0.0):
        return (
            KSAMPLER(
                lying_sigma_sampler,
                extra_options={
                    "lss_wrapped_sampler": sampler,
                    "lss_dishonesty_factor": dishonesty_factor,
                    "lss_startend_percent": (start_percent, end_percent),
                    "lss_fade": fade,
                    "lss_max_step_overshoot": max_step_overshoot,
                },
            ),
        )


class DetailDaemonSamplerGraphNode:
    """Detail Daemon Sampler with Graph Visualization - combines DetailDaemonSamplerNode and DetailDaemonGraphSigmasNode"""
    DESCRIPTION = "This sampler wrapper works by adjusting the sigma passed to the model, while showing a visual graph of the adjustment schedule."
    CATEGORY = "sampling/custom_sampling/samplers"
    RETURN_TYPES = ("SAMPLER",)
    OUTPUT_NODE = True
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "detail_amount": (
                    "FLOAT",
                    {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "start": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "bias": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "exponent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
                "start_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "end_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "fade": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "smooth": ("BOOLEAN", {"default": True}),
                "cfg_scale_override": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "round": 0.01,
                        "tooltip": "If set to 0, the sampler will automatically determine the CFG scale (if possible). Set to some other value to override.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        sampler: object,
        sigmas: torch.Tensor,
        *,
        detail_amount,
        start,
        end,
        bias,
        exponent,
        start_offset,
        end_offset,
        fade,
        smooth,
        cfg_scale_override,
    ) -> tuple:
        # Create the schedule for visualization
        steps = len(sigmas) - 1
        schedule = make_detail_daemon_schedule(
            steps,
            start,
            end,
            bias,
            detail_amount,
            exponent,
            start_offset,
            end_offset,
            fade,
            smooth,
        )

        # Create the graph image
        image = cls.plot_schedule(schedule)
        
        # Save temp image
        output_dir = folder_paths.get_temp_directory()
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(prefix_append, output_dir)
        )
        filename = f"{filename}_{counter:05}_.png"
        file_path = os.path.join(full_output_folder, filename)
        image.save(file_path, compress_level=1)

        # Create the sampler function closure
        def dds_make_schedule(steps):
            return make_detail_daemon_schedule(
                steps,
                start,
                end,
                bias,
                detail_amount,
                exponent,
                start_offset,
                end_offset,
                fade,
                smooth,
            )

        # Return the sampler and UI output
        return {
            "ui": {
                "images": [
                    {"filename": filename, "subfolder": subfolder, "type": "temp"},
                ],
            },
            "result": (
                KSAMPLER(
                    detail_daemon_sampler,
                    extra_options={
                        "dds_wrapped_sampler": sampler,
                        "dds_make_schedule": dds_make_schedule,
                        "dds_cfg_scale_override": cfg_scale_override,
                    },
                ),
            ),
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
        plt.xticks(range(len(schedule)))
        plt.ylim(-1, 1)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG")
        plt.close()
        buf.seek(0)
        image = Image.open(buf)
        return image

