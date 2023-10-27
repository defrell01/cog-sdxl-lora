# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.

import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

pipe = DiffusionPipeline.from_pretrained(
    "jinaai/flat-2d-animerge",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.save_pretrained("./sdxl-cache", safe_serialization=True)

