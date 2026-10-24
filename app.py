import os
import csv
import time
from datetime import datetime

import gradio as gr
import torch
from diffusers import PixArtAlphaPipeline

MODEL_ID = os.getenv("MODEL_ID", "PixArt-alpha/PixArt-XL-2-512x512")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "generation_log.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    print("WARNING: CUDA GPU is not available. Loading on CPU may be extremely slow.")

# Load the pretrained model once when the app starts.
pipe = PixArtAlphaPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True,
    low_cpu_mem_usage=True,
)

if torch.cuda.is_available():
    # Lower GPU memory use. Can be slower than keeping the whole model on GPU.
    pipe.enable_sequential_cpu_offload()

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp", "prompt", "resolution", "steps",
            "seed", "generation_seconds", "model", "image_path"
        ])


def generate_image(prompt, resolution, steps, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter an image prompt.")

    resolution = int(resolution)
    steps = int(steps)
    seed = int(seed)

    if resolution not in (512, 768, 1024):
        raise gr.Error("Choose 512, 768, or 1024 resolution.")
    if not 1 <= steps <= 50:
        raise gr.Error("Inference steps must be between 1 and 50.")
    if not 0 <= seed <= 2**32 - 1:
        raise gr.Error("Seed must be between 0 and 4294967295.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(
        OUTPUT_DIR, f"image_{resolution}_{timestamp}_seed{seed}.png"
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = time.time()

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt.strip(),
                height=resolution,
                width=resolution,
                num_inference_steps=steps,
                guidance_scale=4.5,
                num_images_per_prompt=1,
                generator=generator,
                clean_caption=False,
            )

        image = result.images[0]
        elapsed = time.time() - start
        image.save(image_path)

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                prompt.strip(),
                f"{resolution}x{resolution}",
                steps,
                seed,
                round(elapsed, 2),
                MODEL_ID,
                image_path,
            ])

        return image, (
            f"Success | Resolution: {resolution}x{resolution} | "
            f"Steps: {steps} | Seed: {seed} | Time: {elapsed:.1f}s\n"
            f"Saved image: {image_path}"
        )

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(
            "GPU ran out of memory. Try 512x512 or 768x768, or use a GPU "
            "runtime with more memory. Restart the app only if the session "
            "is no longer responsive."
        )
    except Exception as exc:
        raise gr.Error(f"Generation failed: {type(exc).__name__}: {exc}")


with gr.Blocks(title="AI Image Synthesis — Project 2") as demo:
    gr.Markdown("# AI Image Synthesis Using Diffusion Transformers")
    gr.Markdown(
        "Text-to-image generation using pretrained PixArt-Alpha. "
        "Choose a resolution, inference-step count, and seed."
    )
    prompt = gr.Textbox(
        label="Image prompt",
        placeholder="A futuristic Mumbai skyline at sunset, detailed architecture",
        lines=4,
    )
    with gr.Row():
        resolution = gr.Dropdown(
            choices=["512", "768", "1024"],
            value="768",
            label="Square image resolution",
        )
        steps = gr.Slider(
            minimum=10, maximum=30, value=15, step=5,
            label="Inference steps",
        )
        seed = gr.Number(value=42, precision=0, label="Random seed")
    generate_btn = gr.Button("Generate Image", variant="primary")
    output = gr.Image(label="Generated image", type="pil")
    status = gr.Textbox(label="Generation details", interactive=False)
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, resolution, steps, seed],
        outputs=[output, status],
    )

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
