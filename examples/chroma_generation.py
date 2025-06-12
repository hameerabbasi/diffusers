"""
Example script for generating images with Chroma model
"""

import torch
from diffusers import ChromaTransformer2DModel, ChromaPipeline, AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5TokenizerFast

def generate_with_chroma():
    # Model paths
    chroma_path = "lodestones/Chroma"  # or local path to safetensors
    vae_path = "black-forest-labs/FLUX.1-schnell"  # Chroma uses Flux VAE
    text_encoder_path = "google/t5-v1_1-xxl"  # T5 XXL encoder
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print("Loading models...")
    
    # Load VAE from Flux
    vae = AutoencoderKL.from_pretrained(
        vae_path, 
        subfolder="vae",
        torch_dtype=dtype
    ).to(device)
    
    # Load T5 encoder
    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=dtype
    ).to(device)
    
    tokenizer = T5TokenizerFast.from_pretrained(text_encoder_path)
    
    # Load Chroma transformer
    # Option 1: From HuggingFace Hub (when available)
    # transformer = ChromaTransformer2DModel.from_pretrained(
    #     chroma_path,
    #     torch_dtype=dtype
    # ).to(device)
    
    # Option 2: From single file
    transformer = ChromaTransformer2DModel.from_single_file(
        "path/to/chroma-unlocked-v29.safetensors",
        torch_dtype=dtype
    ).to(device)
    
    # Create scheduler
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    # Create pipeline
    pipe = ChromaPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler
    )
    
    # Enable optimizations
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # Generate image
    prompt = "A majestic lion made of galaxies and stardust, cosmic art style"
    
    print(f"Generating image with prompt: {prompt}")
    
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=20,
        guidance_scale=3.5,
        generator=torch.Generator(device=device).manual_seed(42)
    ).images[0]
    
    # Save image
    image.save("chroma_output.png")
    print("Image saved as chroma_output.png")

if __name__ == "__main__":
    generate_with_chroma()
