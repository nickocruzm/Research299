from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import AutoTokenizer, CLIPTextModel
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class StableDiffusion:
    def __init__(self):
        pass
    
    def load_components(self):
        """
            Loads each individual component that makes up Stable Diffusion v1.5
        """
        model_id = "runwayml/stable-diffusion-v1-5"
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")


sd = StableDiffusion()
sd.load_components()