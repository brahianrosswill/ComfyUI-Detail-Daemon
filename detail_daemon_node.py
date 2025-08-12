# Based on the concept from https://github.com/muerrilla/sd-webui-detail-daemon

from __future__ import annotations

import io

# Trying matplotlib NSWindow warning workaround on macOS
import platform
import math # Added for tanh

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
    artifact_control=0.5, # New parameter with default
):
    # Apply tanh to detail_amount (passed as 'amount') to dampen extreme values.
    # tanh will map 'amount' to a range of -1 to 1.
    # This helps prevent overly aggressive adjustments when detail_amount is very high or low.
    scaled_amount = math.tanh(amount)
    # If the original 'amount' was intended to be used in a larger range,
    # 'scaled_amount' can be further multiplied by a factor here.
    # For now, we'll use the -1 to 1 range directly.

    # Adaptive exponent calculation
    # K is a sensitivity factor for how much scaled_amount affects the exponent.
    # K = 0.75 means at full scaled_amount (1 or -1), exponent is divided by 1.75.
    K = 0.75
    effective_exponent = exponent / (1 + abs(scaled_amount) * K)
    # Ensure the effective_exponent does not become too small to prevent extreme curve shapes or errors.
    effective_exponent = max(0.1, effective_exponent)

    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    # Use effective_exponent instead of the original exponent
    start_values = start_values**effective_exponent
    if start_values.any():
        # Use scaled_amount instead of amount
        start_values *= scaled_amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    # Use effective_exponent instead of the original exponent
    end_values = end_values**effective_exponent
    if end_values.any():
        # Use scaled_amount instead of amount
        end_values *= scaled_amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    # Limit the rate of change between successive multiplier values
    # This helps to smooth out abrupt jumps in the schedule.
    # Controlled by artifact_control: 0.0 = more smoothing (smaller max_delta), 1.0 = less smoothing (larger max_delta)
    base_max_delta = 0.15
    min_max_delta = 0.01
    # artifact_control = 1 -> dynamic_max_delta = base_max_delta
    # artifact_control = 0 -> dynamic_max_delta = min_max_delta
    dynamic_max_delta = min_max_delta + (base_max_delta - min_max_delta) * artifact_control

    if steps > 1:  # Only apply if there's more than one step
        for i in range(1, steps):  # Iterate from the second element
            delta = multipliers[i] - multipliers[i - 1]
            clamped_delta = max(min(delta, dynamic_max_delta), -dynamic_max_delta)
            multipliers[i] = multipliers[i - 1] + clamped_delta

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
                "artifact_control": ( # New input for artifact_control
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Higher values allow more schedule variance (less smoothing), lower values increase smoothing (more artifact control)."}
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
        artifact_control, # Added artifact_control parameter
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
            artifact_control, # Pass artifact_control
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
    sigmas: torch.Tensor, # Should be sigmas_cpu
    dd_schedule: torch.Tensor, # Should be on CPU
) -> float:
    sched_len = len(dd_schedule)
    if (
        sched_len < 2
        or len(sigmas) < 2
        or sigma <= 0
        or not (sigmas[-1] <= sigma <= sigmas[0]) # sigmas is expected to be sorted descending
    ):
        return 0.0
    # First, we find the index of the closest sigma in the list to what the model was
    # called with.
    # sigmas is expected to be a 1D CPU tensor. sigma is a float.
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin()) # Convert tensor index to int

    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2]) # sigmas[-2] because sigmas has one more element than schedule usually
        or deltas[idx] == 0
    ):
        # Either exact match or closest to head/tail of the DD schedule so we
        # can't interpolate to another schedule item.
        return dd_schedule[idx].item() # dd_schedule is a CPU tensor

    # If we're here, that means the sigma is in between two sigmas in the list.
    # Original logic for determining idxlow and idxhigh based on int idx
    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)

    # We find the low/high neighbor sigmas - our sigma is somewhere between them.
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh] # These are tensor elements (0-dim tensors)

    if nhigh - nlow == 0:
        # Shouldn't be possible if sigmas are distinct, but just in case... Avoid divide by zero.
        return dd_schedule[idxlow].item() # Return item from dd_schedule at idxlow

    # Ratio of how close we are to the high neighbor.
    # sigma is float, nlow, nhigh are 0-dim tensors; operations will promote sigma to tensor.
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)

    # Mix the DD schedule high/low items according to the ratio.
    # dd_schedule elements are 0-dim tensors. lerp works with these.
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
        device="cpu",  # Revert to CPU device
    )
    sigmas_cpu = sigmas.detach().clone().cpu() # Reintroduce sigmas_cpu
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05 # Use sigmas_cpu

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict):
        sigma_float = float(sigma.max().detach().cpu()) # Reintroduce sigma_float
        if not (sigma_min <= sigma_float <= sigma_max): # Use sigma_float for comparison
            return model(x, sigma, **extra_args)
        # Call original get_dd_schedule with sigma_float, sigmas_cpu, and CPU dd_schedule
        dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        adjusted_sigma = sigma * max(0.001, 1.0 - dd_adjustment * cfg_scale)
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
                "artifact_control": ( # New input for artifact_control
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Higher values allow more schedule variance (less smoothing), lower values increase smoothing (more artifact control)."}
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
        artifact_control, # Added artifact_control parameter
    ) -> tuple:
        def dds_make_schedule(steps): # Closure captures artifact_control from outer scope
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
                artifact_control, # Pass artifact_control
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


def detail_daemon_wan21_sampler(
    # This function is intended for the "wan 2.1" video generator.
    # It adapts the detail daemon logic for video generation.
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    dds_wrapped_sampler: object,
    dds_make_schedule: callable,
    # dds_cfg_scale_override: float, # This parameter is removed
    **kwargs: dict,
) -> torch.Tensor:
    # wan 2.1 might have a different way to access CFG scale or model configuration;
    # this might need adjustment.
    # For now, we exclusively use the model's CFG scale.
    maybe_cfg_scale = getattr(model.inner_model, "cfg", None)
    cfg_scale = (
        float(maybe_cfg_scale) if isinstance(maybe_cfg_scale, (int, float)) else 1.0
    )
    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",  # Revert to CPU device
    )
    sigmas_cpu = sigmas.detach().clone().cpu() # Reintroduce sigmas_cpu
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05 # Use sigmas_cpu

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict):
        sigma_float = float(sigma.max().detach().cpu()) # Reintroduce sigma_float
        if not (sigma_min <= sigma_float <= sigma_max): # Use sigma_float for comparison
            return model(x, sigma, **extra_args)
        # Call original get_dd_schedule with sigma_float, sigmas_cpu, and CPU dd_schedule
        dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        adjusted_sigma = sigma * max(0.001, 1.0 - dd_adjustment * cfg_scale)
        # wan 2.1 model interaction: Ensure 'model', 'x', and 'adjusted_sigma' are
        # compatible with wan 2.1's API. Specific conditioning or attributes
        # for wan 2.1 might need to be handled here.
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


class DetailDaemonWan21SamplerNode:
    DESCRIPTION = "This sampler wrapper is adapted for the wan 2.1 video generator. It adjusts sigmas based on a schedule to control detail."
    CATEGORY = "sampling/custom_sampling/samplers/wan2.1"
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
                "artifact_control": ( # New input for artifact_control
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Higher values allow more schedule variance (less smoothing), lower values increase smoothing (more artifact control)."}
                ),
                # "cfg_scale_override" is removed from here
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
        artifact_control, # Added artifact_control parameter
        # cfg_scale_override, # This parameter is removed
    ) -> tuple:
        def dds_make_schedule(steps): # Closure captures artifact_control from outer scope
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
                artifact_control, # Pass artifact_control
            )

        return (
            KSAMPLER(
                detail_daemon_wan21_sampler,
                extra_options={
                    "dds_wrapped_sampler": sampler,
                    "dds_make_schedule": dds_make_schedule,
                    # "dds_cfg_scale_override": cfg_scale_override, # This is removed
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

