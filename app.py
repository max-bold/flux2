import torch
import gradio as gr
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
import gc
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np

# Initialize model
device = "cuda"
dtype = torch.bfloat16

print("Loading model...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype
)
# Enable CPU offload for memory efficiency
pipe.enable_model_cpu_offload()
print("Model loaded!")

# Create generator once and reuse
generator = torch.Generator(device=device)


def generate_image(
    prompt,
    base_image,
    height,
    width,
    num_inference_steps,
    progress=None,
):
    if progress is None:
        progress = gr.Progress()

    if base_image is None:
        return None
    if not prompt.strip():
        return None

    result_img = None
    out = None
    try:
        orig_width, orig_height = base_image.size
        orig_aspect_ratio = orig_width / orig_height

        height = int(height)
        width = int(width)
        num_inference_steps = int(num_inference_steps)

        new_height = int(width / orig_aspect_ratio)
        new_height = (new_height // 16) * 16
        width = (width // 16) * 16
        new_height = max(new_height, 256)
        width = max(width, 256)

        if base_image.mode != "RGB":
            base_image = base_image.convert("RGB")

        progress(0, desc="Preparing...")

        def progress_callback(
            pipe, step: int, timestep: int, callback_kwargs: dict
        ) -> dict:
            progress(
                (step + 1) / num_inference_steps,
                desc=f"Generating... Step {step+1}/{num_inference_steps}",
            )
            # на всякий: не тащим дальше ничего тяжёлого
            callback_kwargs.pop("latents", None)
            return callback_kwargs

        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                image=base_image,
                height=new_height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback_on_step_end=progress_callback, #type: ignore
                # УБРАЛИ: callback_on_step_end_tensor_inputs=["latents"],
                guidance_scale=1.0,
            )

        result_img = out.images[0]  # type: ignore
        return result_img

    except Exception as e:
        return None

    finally:
        # разрываем ссылки на крупные объекты
        try:
            del out
        except Exception:
            pass
        try:
            del base_image
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


def edit_image(generated_image):
    """
    Transfers the generated image to base image field for editing
    """
    if generated_image is None:
        return "", None

    return "", generated_image


def save_image(image):
    """
    Save image with dialog for format and location selection
    """
    if image is None:
        return "No image to save"
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            # If float, scale to 0-255
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")]
    )
    
    if file_path:
        try:
            image.save(file_path)
            return f"✅ Image saved to {file_path}"
        except Exception as e:
            return f"❌ Error saving image: {str(e)}"
    else:
        return "Cancel"


def update_height(base_image, width):
    """
    Updates height to maintain aspect ratio when width changes
    """
    if base_image is None or width is None:
        return 576

    try:
        orig_width, orig_height = base_image.size
        orig_aspect_ratio = orig_width / orig_height
        width = int(width)

        new_height = int(width / orig_aspect_ratio)
        new_height = (new_height // 16) * 16

        if new_height < 256:
            new_height = 256

        return float(new_height)
    except:
        return 576


# Create Gradio interface
with gr.Blocks(title="Flux2 Klein Generator") as demo:
    gr.Markdown("# 🎨 Flux2 Klein Image Generator")
    gr.Markdown("Upload a base image and enter a prompt to generate new images")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="📝 Prompt",
                placeholder="Describe what you want to generate...",
                lines=3,
            )
            image_input = gr.Image(label="🖼️ Base Image", type="pil")

            with gr.Row():
                height_input = gr.Number(
                    label="📏 Height (ignored, auto-calculated)",
                    value=576,
                    interactive=False,
                )
                width_input = gr.Number(
                    label="📐 Width (reference)",
                    value=1024,
                )

            num_inference_steps_input = gr.Slider(
                label="⚙️ Inference Steps",
                minimum=10,
                maximum=100,
                step=1,
                value=50,
            )

            generate_btn = gr.Button("🚀 Generate", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="📸 Result", buttons=['download', 'fullscreen'], interactive=False)
            edit_btn = gr.Button("✏️ Edit Image", variant="secondary")
            save_btn = gr.Button("💾 Save Image", variant="secondary")
            save_status = gr.Textbox(label="Status", interactive=False, value="")

    # Connect generate button
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            image_input,
            height_input,
            width_input,
            num_inference_steps_input,
        ],
        outputs=[image_output],
    )

    # Update height when width changes (preserve aspect ratio)
    width_input.change(
        fn=update_height,
        inputs=[image_input, width_input],
        outputs=[height_input],
    )

    # Update height when base image changes
    image_input.change(
        fn=update_height,
        inputs=[image_input, width_input],
        outputs=[height_input],
    )

    # Connect edit button
    edit_btn.click(
        fn=edit_image,
        inputs=[image_output],
        outputs=[prompt_input, image_input],
    )

    # Connect save button
    save_btn.click(
        fn=save_image,
        inputs=[image_output],
        outputs=[save_status],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
