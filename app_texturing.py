import gradio as gr

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime
import shutil
from typing import *
import torch
import numpy as np
import trimesh
import aspose.threed as a3d
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopoDS import TopoDS_Shape
import trimesh
import os
from PIL import Image
from modelscope import snapshot_download
from trellis2.pipelines import Trellis2TexturingPipeline
from transformers import AutoTokenizer, CLIPTextModel
from modelscope import ZImagePipeline

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

# Text conditioning model (initialized lazily)
text_cond_model = None
text_tokenizer = None

# Text-to-Image pipeline (initialized lazily)
text2img_pipeline = None

def get_text2img_pipeline():
    """Lazy initialization of text-to-image pipeline."""
    global text2img_pipeline
    if text2img_pipeline is None:
        # Use Stable Diffusion XL from ModelScope
        print("Loading Stable Diffusion pipeline...")
        text2img_pipeline = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        text2img_pipeline.to("cuda")
        # text2img_pipeline = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16)
        # Enable CPU offloading to save VRAM - model will be moved to GPU only during inference
        # text2img_pipeline.enable_model_cpu_offload()
        print("SDXL pipeline loaded successfully!")
    return text2img_pipeline

@torch.no_grad()
def generate_image_from_text(prompt: str, seed: int = 42) -> Image.Image:
    """Generate image from text prompt using Z-Image, then unload to free GPU memory."""
    global text2img_pipeline

    # Load pipeline
    pipe = get_text2img_pipeline()
    aspect_ratios = {
        "1:1": (1024, 1024),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios["1:1"]
    negative_prompt = ""  # Optional, but would be powerful when you want to remove some unwanted content
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=width,
        width=height,
        cfg_normalization=False,
        num_inference_steps=50,
        guidance_scale=4,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    image.save("output_t2i.png")
    # Unload pipeline to free GPU memory for TRELLIS
    print("Unloading Z-Image pipeline to free GPU memory...")
    del text2img_pipeline
    text2img_pipeline = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Z-Image pipeline unloaded. GPU memory freed.")
    image = preprocess_image(image)
    return image


def _init_text_cond_model():
    """
    Initialize the text conditioning model.
    """
    global text_cond_model
    if text_cond_model is not None:
        return
    model_dir = snapshot_download('muse/openai-clip-vit-large-patch14')
    # load model
    model = CLIPTextModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    model = model.cuda()
    text_cond_model = {
        'model': model,
        'tokenizer': tokenizer,
    }
    text_cond_model['null_cond'] = encode_text([''])


@torch.no_grad()
def encode_text(text: List[str]) -> torch.Tensor:
    """
    Encode the text.
    """
    assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
    encoding = text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True,
                                                 return_tensors='pt')
    tokens = encoding['input_ids'].cuda()
    embeddings = text_cond_model['model'](input_ids=tokens).last_hidden_state

    return embeddings


def get_cond(prompt: List[str]) -> dict:
    """
    Get the conditioning information for the model.

    Args:
        prompt (List[str]): The text prompt.

    Returns:
        dict: The conditioning information
    """
    _init_text_cond_model()
    cond = encode_text(prompt)
    neg_cond = text_cond_model['null_cond']
    return {
        'cond': cond,
        'neg_cond': neg_cond,
    }

def get_text_cond_model():
    """Lazy initialization of text conditioning model."""
    global text_cond_model, text_tokenizer
    if text_cond_model is None:
        # 从 ModelScope 下载 CLIP 模型
        model_dir = snapshot_download('AI-ModelScope/clip-vit-large-patch14')
        text_cond_model = CLIPTextModel.from_pretrained(model_dir).cuda()
        text_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        text_cond_model.eval()
    return text_cond_model, text_tokenizer

@torch.no_grad()
def encode_text_prompt(text: str) -> torch.Tensor:
    """Encode text prompt using CLIP."""
    if not text or text.strip() == "":
        return None
    model, tokenizer = get_text_cond_model()
    encoding = tokenizer([text], max_length=77, padding='max_length', truncation=True, return_tensors='pt')
    tokens = encoding['input_ids'].cuda()
    embeddings = model(input_ids=tokens).last_hidden_state
    return embeddings


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def step_to_glb_via_stl(step_file, glb_file):
    """
    通过 STL 中间格式转换 STEP 到 GLB
    """
    # 1. 读取 STEP 文件
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)

    if status != IFSelect_RetDone:
        print("无法读取 STEP 文件")
        return False

    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    # 2. 转换为 STL
    stl_file = "temp.stl"

    # 创建网格
    mesh = BRepMesh_IncrementalMesh(shape, 0.01, True)
    mesh.Perform()

    # 写入 STL
    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape, stl_file)

    # 3. STL 转 GLB
    mesh_trimesh = trimesh.load(stl_file)

    # 保存为 GLB
    mesh_trimesh.export(glb_file)

    # 清理临时文件
    if os.path.exists(stl_file):
        os.remove(stl_file)

    print(f"转换完成: {glb_file}")
    return True


def shapeimage_to_tex(
        mesh_file: str,
        image: Image.Image,
        text_prompt: str,
        seed: int,
        resolution: str,
        texture_size: int,
        tex_slat_guidance_strength: float,
        tex_slat_guidance_rescale: float,
        tex_slat_sampling_steps: int,
        tex_slat_rescale_t: float,
        req: gr.Request,
        progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str, Image.Image]:
    if mesh_file.lower().endswith('.jt'):
        user_dir = os.path.join(TMP_DIR, str(req.session_hash))
        os.makedirs(user_dir, exist_ok=True)
        glb_temp_path = os.path.join(user_dir, os.path.basename(mesh_file).replace('.jt', '.glb'))
        scene = a3d.Scene.from_file(mesh_file)
        scene.save(glb_temp_path)
        mesh_file = glb_temp_path
    elif mesh_file.lower().endswith(('.stp', '.step')):
        user_dir = os.path.join(TMP_DIR, str(req.session_hash))
        os.makedirs(user_dir, exist_ok=True)
        ext = os.path.splitext(mesh_file)[1]
        glb_temp_path = os.path.join(user_dir, os.path.basename(mesh_file).replace(ext, '.glb'))
        step_to_glb_via_stl(mesh_file, glb_temp_path)
        mesh_file = glb_temp_path

    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    # Check if we need to generate image from text
    generated_img = None
    if (text_prompt and text_prompt.strip()) and image is None:
        # Text-only mode: generate image from text first
        print(f"Generating image from text prompt: {text_prompt}")
        image = generate_image_from_text(text_prompt, seed)
        generated_img = image.copy()
        print("Image generated successfully")

    output = pipeline.run(
        mesh,
        image,
        seed=seed,
        preprocess_image=True,  # Enable preprocessing for generated images
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        resolution=int(resolution),
        texture_size=texture_size,
    )
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    output.export(glb_path, extension_webp=False)
    torch.cuda.empty_cache()
    return glb_path, glb_path, generated_img


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Texturing a mesh with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2)
    * Upload a mesh and provide either:
      - A reference image (preferably with an alpha-masked foreground object), or
      - A text prompt (will use SDXL to generate an image first, then create texture)
    * Click Generate to create a textured 3D asset.
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            mesh_file = gr.File(label="Upload Mesh",
                                file_types=[".ply", ".obj", ".glb", ".gltf", ".jt", ".stp", ".step"],
                                file_count="single")
            image_prompt = gr.Image(label="Image Prompt (Optional)", format="png", image_mode="RGBA", type="pil", height=400)
            text_prompt = gr.Textbox(label="Text Prompt (Optional - Will generate image using SDXL if no image provided)", placeholder="Enter text description to generate texture...", lines=2)

            resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)

            generate_btn = gr.Button("Generate")

            with gr.Accordion(label="Advanced Settings", open=False):
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)

        with gr.Column(scale=10):
            generated_image = gr.Image(label="Generated Image (from text prompt)", format="png", height=300, visible=True)
            glb_output = gr.Model3D(label="Extracted GLB", height=724, show_label=True, display_mode="solid",
                                    clear_color=(0.25, 0.25, 0.25, 1.0))
            download_btn = gr.DownloadButton(label="Download GLB")

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        shapeimage_to_tex,
        inputs=[
            mesh_file, image_prompt, text_prompt, seed, resolution, texture_size,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t,
        ],
        outputs=[glb_output, download_btn, generated_image],
    )

# Launch the Gradio app
if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    # 指定模型ID，会从ModelScope镜像下载
    model_dir = snapshot_download('microsoft/TRELLIS.2-4B')
    print(f'模型已下载到: {model_dir}')
    pipeline = Trellis2TexturingPipeline.from_pretrained(model_dir, config_file="texturing_pipeline.json")
    pipeline.cuda()

    demo.launch(server_name="0.0.0.0", server_port=8889)
