# AI Image Synthesis Using Diffusion Transformers

A text-to-image demo built with a pretrained PixArt-Alpha pipeline and a Gradio interface. It supports square outputs at 512, 768, and 1024 pixels, adjustable inference steps, seed selection, image saving, and CSV experiment logging.

## Current implementation

- Pretrained model: `PixArt-alpha/PixArt-XL-2-512x512`
- Interface: Gradio
- Pipeline: Hugging Face Diffusers
- Outputs: PNG images and `outputs/generation_log.csv`
- The app uses sequential CPU offloading when CUDA is available to reduce GPU memory use.

## Requirements

- Python 3.10+ recommended
- A CUDA-capable GPU strongly recommended
- Enough RAM and disk space to download/load the model
- Internet access for the initial model download

Free GPU availability and runtime limits depend on the platform. The model is large and may not run comfortably on CPU-only hosting.

## Run locally or in a GPU notebook

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   python app.py
   ```

4. Open the local Gradio URL shown in the terminal.

The first launch downloads the model checkpoint and may take time. To use a different compatible model, set the `MODEL_ID` environment variable.

## Controls

- **Prompt:** describes the desired image.
- **Resolution:** 512, 768, or 1024 square output.
- **Inference steps:** more steps can take longer; quality does not improve monotonically.
- **Seed:** controls the random starting noise and can help reproduce generations in a compatible environment.

## Outputs and logs

The app saves PNGs and appends metadata to `outputs/generation_log.csv`. These files are stored on the machine/container running the app. Ephemeral notebook or hosting storage may be cleared, so download or back up important results.

## Architecture

```text
Text prompt + settings
        |
        v
Gradio interface
        |
        v
Pretrained PixArt-Alpha text-to-image pipeline
        |
        v
Generated image (PNG) + CSV metadata log
```

## Research status and limitations

This is the pretrained-generator phase (Part A). It is not a model trained from scratch, and it does not yet implement the proposed RAE-versus-VAE research comparison. Do not report comparative research results until those experiments have actually been run.

Higher output resolution does not guarantee greater semantic accuracy or proportionally more real detail. Results depend on prompt, model, settings, and available compute.

## License and model usage

Review the PixArt-Alpha model card and license/usage terms before redistribution or public deployment. This repository contains application code, not model weights.

## Running the Demo on Kaggle

The project was tested in a Kaggle Notebook with Internet enabled and two NVIDIA Tesla T4 GPUs.

### Setup
1. Install the required Python libraries using the provided `requirements.txt` or the installation command documented in the notebook.
2. Load the pretrained PixArt-Alpha model: `PixArt-alpha/PixArt-XL-2-512x512`.
3. Generate a test image at 512 × 512 resolution.
4. Launch the Gradio interface using `demo.launch(share=True)`.

### Features tested
- Text-to-image generation from a user prompt
- Resolution selection
- Inference-step and random-seed controls
- Image output through a Gradio interface

### Limitations
The public Gradio link is temporary and depends on the Kaggle session remaining active. GPU availability and usage are subject to Kaggle's limits. The current implementation uses a pretrained PixArt-Alpha model; it does not yet include a completed RAE-versus-VAE research comparison.

