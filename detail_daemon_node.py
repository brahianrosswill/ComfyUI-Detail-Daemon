import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as T
import comfy.samplers

class DetailDaemonNode:
    @classmethod
    def INPUT_TYPES(s):
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
                "mode": (["cond", "uncond", "both"],),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
            }
        }

    RETURN_TYPES = ("SIGMAS", "IMAGE",)
    RETURN_NAMES = ("adjusted_sigmas", "plot_adjustment_image",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    FUNCTION = "adjust_sigmas"

    def adjust_sigmas(self, sigmas, detail_amount, start, end, bias, exponent, start_offset, end_offset, fade, smooth, mode, cfg_scale, sampler_name):
           
        # Create a copy of the input sigmas using clone() for tensors to avoid modifying the original
        sigmas = sigmas.clone()

        # Derive the number of steps from the length of sigmas minus 1 (ignore the final sigma)
        steps = len(sigmas) - 1  # 21 sigmas, 20 steps
        
        if sampler_name in ['dpmpp_sde', 'dpmpp_2s_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral']:
            actual_steps = steps * 2 - 1
        else:
            actual_steps = steps

        # Create the schedule using the number of steps
        schedule = self.make_schedule(
            actual_steps, start, end, bias, detail_amount, exponent, start_offset, end_offset, fade, smooth
        )

        # Debugging: print schedule and sigmas lengths to verify alignment
        print(f"Number of sigmas: {len(sigmas)}, Number of schedule steps: {len(schedule)}")

        # Iterate over the sigmas, except for the last one (which we assume is 0 and leave untouched)
        for idx in range(steps):
            multiplier = schedule[idx] * 0.1

            # Debugging: print each index and sigma to track what's being adjusted
            print(f"Adjusting sigma at index {idx} with multiplier {multiplier}")

            # Adjust sigma values based on mode
            if sigmas.shape[0] == 1:  # Default to "both" mode if sigma size is 1
                mode = "both"
                if idx == 0:
                    print("WARNING: Defaulting to 'both' mode because cond/uncond are not supported with this sampler.")
            
            if mode == "cond":
                sigmas[idx] *= (1 - multiplier)  # Adjust each sigma value in "cond" mode
            elif mode == "uncond":
                sigmas[idx] *= (1 + multiplier)  # Adjust each sigma value in "uncond" mode
            else:  # both
                sigmas[idx] *= (1 - multiplier * cfg_scale)  # Adjust each sigma in "both" mode
                
        # Create the plot for visualization
        image = self.plot_schedule(schedule)
        
        return (sigmas, image)

    def make_schedule(self, steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
        start = min(start, end)
        mid = start + bias * (end - start)
        multipliers = np.zeros(steps)

        start_idx, mid_idx, end_idx = [int(round(x * (steps - 1))) for x in [start, mid, end]]        

        start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
        if smooth:  
            start_values = 0.5 * (1 - np.cos(start_values * np.pi))
        start_values = start_values ** exponent
        if start_values.any():
            start_values *= (amount - start_offset)  
            start_values += start_offset  

        end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
        if smooth:
            end_values = 0.5 * (1 - np.cos(end_values * np.pi))
        end_values = end_values ** exponent
        if end_values.any():
            end_values *= (amount - end_offset)  
            end_values += end_offset  

        multipliers[start_idx:mid_idx+1] = start_values
        multipliers[mid_idx:end_idx+1] = end_values        
        multipliers[:start_idx] = start_offset
        multipliers[end_idx+1:] = end_offset    
        multipliers *= 1 - fade

        return multipliers

    def plot_schedule(self, schedule):
        plt.figure(figsize=(6, 4))  # Adjusted width
        plt.plot(schedule, label='Sigma Adjustment Curve')
        plt.xlabel('Steps')
        plt.ylabel('Multiplier (*10)')
        plt.title('Detail Adjustment Schedule')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(len(schedule)))
        plt.ylim(-1, 1)

        # Use tight_layout or subplots_adjust
        plt.tight_layout()
        # Or manually adjust if needed:
        # plt.subplots_adjust(left=0.2)

        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        plt.close()
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = T.ToTensor()(image).permute(1, 2, 0).unsqueeze(0)
        return image_tensor

class MultiplySigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "factor": ("FLOAT", {"default": 1, "min": 0, "max": 100, "step": 0.001}) 
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def simple_output(self, sigmas, factor):
        # Clone the sigmas to ensure the input is not modified (stateless)
        sigmas = sigmas.clone()
        return (sigmas * factor,)
