import os
import gc
import random
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import Flux2KleinPipeline
from diffusers.quantizers.quantization_config import TorchAoConfig
from viso_bridge import VisoBridge, RESTORER_CHOICES

# Monkeypatch TorchAoConfig.from_dict to handle string quant_type
original_from_dict = TorchAoConfig.from_dict
@classmethod
def patched_from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
    if "quant_type" in config_dict and isinstance(config_dict["quant_type"], str):
        qt = config_dict["quant_type"]
        if qt == "float8wo":
            config_dict["quant_type"] = {
                "default": {
                    "_type": "Float8WeightOnlyConfig",
                    "_version": 2,
                    "_data": {
                        "weight_dtype": {"_type": "torch.dtype", "_data": "float8_e4m3fn"},
                        "set_inductor_config": True
                    }
                }
            }
        elif qt == "int8wo":
            config_dict["quant_type"] = {
                "default": {
                    "_type": "Int8WeightOnlyConfig",
                    "_version": 2,
                    "_data": {}
                }
            }
    return original_from_dict(config_dict, return_unused_kwargs=return_unused_kwargs, **kwargs)
TorchAoConfig.from_dict = patched_from_dict

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
MAX_SEED = np.iinfo(np.int32).max

# Model
REPO_ID_4B = "tonera/FLUX.2-klein-4B-fp8-diffusers"

LORA_DIR = os.path.join(os.path.dirname(__file__), "BFS-Best-Face-Swap")

# List LoRAs
available_loras = [f for f in os.listdir(LORA_DIR) if f.endswith(".safetensors") and "4b" in f.lower()]
default_lora = "bfs_head_v1_flux-klein_4b.safetensors" if "bfs_head_v1_flux-klein_4b.safetensors" in available_loras else (available_loras[0] if available_loras else None)

# Global variables to store pipelines and viso bridge
pipe = None
current_lora = None
current_model = None
viso = None

def load_flux_model(model_id, lora_filename):
    global pipe, current_lora, current_model
    if pipe is not None and current_model == model_id and current_lora == lora_filename:
        return pipe
    
    print(f"Loading Flux model {model_id}...")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    
    if lora_filename:
        lora_path = os.path.join(LORA_DIR, lora_filename)
        print(f"Loading LoRA from {lora_path}...")
        pipe.load_lora_weights(lora_path)
        current_lora = lora_filename
    
    current_model = model_id
    return pipe

def get_viso():
    global viso
    if viso is None:
        print("Initializing Viso Engine...")
        viso = VisoBridge(device=device)
    return viso

def flux_face_swap(
    reference_face: Image.Image,
    target_image: Image.Image,
    model_id: str,
    lora_filename: str,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    progress=gr.Progress(track_tqdm=True)
):
    if reference_face is None or target_image is None:
        raise gr.Error("Please provide both a reference face and a target image!")

    load_flux_model(model_id, lora_filename)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)

    # Inverted order as per BFS README: Body first, then Face
    image_list = [target_image, reference_face]

    progress(0.2, desc="Generating...")
    
    image = pipe(
        prompt=prompt,
        image=image_list,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return image, seed

def unload_flux_model():
    global pipe, current_lora, current_model
    if pipe is not None:
        print("Unloading Flux model...")
        pipe.to("cpu") 
        del pipe
        pipe = None
        current_lora = None
        current_model = None
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("VRAM cleared.")

def unload_viso_engine():
    global viso
    if viso is not None:
        print("Unloading Viso Engine...")
        del viso
        viso = None
        
        gc.collect()
        torch.cuda.empty_cache()
        print("Viso VRAM cleared.")

def viso_swap_image(source_img, target_img, model_name, restorer_type, restorer_blend, fidelity_weight):
    v = get_viso()
    return v.process_image(
        source_img, target_img,
        swapper_model=model_name,
        restorer_type=restorer_type,
        restorer_blend=int(restorer_blend),
        fidelity_weight=float(fidelity_weight),
    )

def viso_swap_video(source_img, target_video, model_name, restorer_type, restorer_blend, fidelity_weight, progress=gr.Progress()):
    if source_img is None or target_video is None:
        raise gr.Error("Please provide both a source image and a target video!")
    
    v = get_viso()
    output_path = "output_video.mp4"
    v.process_video(
        source_img, target_video, output_path,
        swapper_model=model_name,
        restorer_type=restorer_type,
        restorer_blend=int(restorer_blend),
        fidelity_weight=float(fidelity_weight),
        progress=progress,
    )
    return output_path

# UI
FACE_SWAP_PROMPT = """head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. Remove the head from Picture 1 completely and replace it with the head from Picture 2.

FROM PICTURE 1 (strictly preserve):
- Scene: lighting conditions, shadows, highlights, color temperature, environment, background
- Head positioning: exact rotation angle, tilt, direction the head is facing
- Expression: facial expression, micro-expressions, eye gaze direction, mouth position, emotion

FROM PICTURE 2 (strictly preserve identity):
- Facial structure: face shape, bone structure, jawline, chin
- All facial features: eye color, eye shape, nose structure, lip shape and fullness, eyebrows
- Hair: color, style, texture, hairline
- Skin: texture, tone, complexion

The replaced head must seamlessly match Picture 1's lighting and expression while maintaining the complete identity from Picture 2. High quality, photorealistic, sharp details, 4k."""

with gr.Blocks(title="BHS Ultimate HeadSwap") as demo:
    gr.Markdown("# 🧬 BHS Ultimate HeadSwap & FaceSwap")
    
    with gr.Tabs():
        with gr.TabItem("Flux High-Fidelity (Head Swap)") as flux_tab:
            with gr.Row():
                with gr.Column():
                    flux_ref_face = gr.Image(label="Reference Face (Identity)", type="pil")
                    flux_target_img = gr.Image(label="Target Image (Body/Scene)", type="pil")
                    flux_lora = gr.Dropdown(choices=available_loras, value=default_lora, label="BFS LoRA")
                    flux_prompt = gr.Textbox(lines=5, value=FACE_SWAP_PROMPT, label="Prompt")
                    
                    with gr.Accordion("Settings", open=False):
                        flux_seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                        flux_rand_seed = gr.Checkbox(label="Randomize Seed", value=True)
                        flux_width = gr.Slider(label="Width", minimum=256, maximum=1024, step=8, value=1024)
                        flux_height = gr.Slider(label="Height", minimum=256, maximum=1024, step=8, value=1024)
                        flux_steps = gr.Slider(label="Steps", minimum=1, maximum=20, step=1, value=4)
                        flux_cfg = gr.Slider(label="Guidance", minimum=0.0, maximum=5.0, step=0.1, value=1.0)
                    
                    flux_btn = gr.Button("Generate Head Swap", variant="primary")
                
                with gr.Column():
                    flux_output = gr.Image(label="Result")
                    flux_seed_out = gr.Number(label="Seed Used", visible=False)
            
            flux_btn.click(
                fn=flux_face_swap,
                inputs=[flux_ref_face, flux_target_img, gr.State(REPO_ID_4B), flux_lora, flux_prompt, flux_seed, flux_rand_seed, flux_width, flux_height, flux_steps, flux_cfg],
                outputs=[flux_output, flux_seed_out]
            )
            pass

        with gr.TabItem("Viso Fast Swap (Image)") as viso_image_tab:
            with gr.Row():
                with gr.Column():
                    viso_source = gr.Image(label="Source Face", type="pil")
                    viso_target = gr.Image(label="Target Image", type="pil")
                    viso_model = gr.Dropdown(
                        choices=["Inswapper128", "SimSwap512", "GhostFace-v1", "GhostFace-v2", "GhostFace-v3", "CSCS"],
                        value="Inswapper128",
                        label="ONNX Swap Model"
                    )
                    with gr.Accordion("Enhancement (improves face quality)", open=False):
                        viso_restorer = gr.Dropdown(
                            choices=RESTORER_CHOICES,
                            value="GFPGAN-v1.4",
                            label="Face Restorer",
                            info="Run a restorer on each swapped face to sharpen details"
                        )
                        viso_blend = gr.Slider(
                            minimum=0, maximum=100, step=1, value=80,
                            label="Restorer Blend %",
                            info="100 = full restorer, 0 = swap only"
                        )
                        viso_fidelity = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.75,
                            label="Fidelity Weight (CodeFormer)",
                            info="Higher = more faithful to input identity"
                        )
                    viso_btn = gr.Button("Fast Swap Image", variant="primary")
                with gr.Column():
                    viso_output = gr.Image(label="Result")
            
            viso_btn.click(
                fn=viso_swap_image,
                inputs=[viso_source, viso_target, viso_model, viso_restorer, viso_blend, viso_fidelity],
                outputs=[viso_output]
            )
            pass

        with gr.TabItem("Viso Fast Swap (Video)") as viso_video_tab:
            with gr.Row():
                with gr.Column():
                    viso_v_source = gr.Image(label="Source Face", type="pil")
                    viso_v_target = gr.Video(label="Target Video")
                    viso_v_model = gr.Dropdown(
                        choices=["Inswapper128", "SimSwap512", "GhostFace-v1", "GhostFace-v2", "GhostFace-v3", "CSCS"],
                        value="Inswapper128",
                        label="ONNX Swap Model"
                    )
                    with gr.Accordion("Enhancement (improves face quality)", open=False):
                        viso_v_restorer = gr.Dropdown(
                            choices=RESTORER_CHOICES,
                            value="GFPGAN-v1.4",
                            label="Face Restorer",
                            info="Run a restorer on each swapped face per frame"
                        )
                        viso_v_blend = gr.Slider(
                            minimum=0, maximum=100, step=1, value=80,
                            label="Restorer Blend %",
                            info="100 = full restorer, 0 = swap only"
                        )
                        viso_v_fidelity = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.75,
                            label="Fidelity Weight (CodeFormer)",
                            info="Higher = more faithful to input identity"
                        )
                    viso_v_btn = gr.Button("Fast Swap Video", variant="primary")
                with gr.Column():
                    viso_v_output = gr.Video(label="Result Video")
                    vram_status = gr.Textbox(visible=False)
            viso_v_btn.click(
                fn=viso_swap_video,
                inputs=[viso_v_source, viso_v_target, viso_v_model, viso_v_restorer, viso_v_blend, viso_v_fidelity],
                outputs=[viso_v_output]
            )
            pass

    viso_image_tab.select(
        fn=unload_flux_model,
        inputs=[],
        outputs=[vram_status]
    )

    viso_video_tab.select(
        fn=unload_flux_model,
        inputs=[],
        outputs=[vram_status]
    )

    flux_tab.select(
        fn=unload_viso_engine,
        inputs=[],
        outputs=[vram_status]
    )

    gr.Markdown("---")
    gr.Markdown("Based on BFS - Best Face Swap & VisoMaster")

if __name__ == "__main__":
    demo.launch(share=False)
