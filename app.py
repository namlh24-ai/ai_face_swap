import cv2
import numpy as np
from PIL import Image
import gradio as gr
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer
from models.model_downloader import ensure_models
import os

# Bước 1: Tải/check models
model_dir = ensure_models()  # Đảm bảo models/ chứa inswapper_128.onnx và buffalo_l/
# InsightFace yêu cầu root chứa thư mục 'models/'. Nếu model_dir = 'models', root phải là '.'
insight_root = os.path.abspath(os.path.join(model_dir, os.pardir))
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], root=insight_root)
app.prepare(ctx_id=0, det_size=(640, 640))

# GFPGAN cho enhance
restorer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None)

# InSwapper
inswapper = get_model(os.path.join(model_dir, 'inswapper_128.onnx'), providers=['CPUExecutionProvider'])

def load_image(image):
    """Load từ Gradio Image (PIL) sang OpenCV."""
    if image is None:
        return None
    img_rgb = np.array(image)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def swap_faces(source_img, template_img, multi_sources=None):
    """
    Swap logic: Single/multi source vào template (auto-multi-face).
    multi_sources: List PIL images từ Gradio folder.
    """
    if source_img is None or template_img is None:
        return None, None, None, "Vui lòng upload source và template!"

    source_cv = load_image(source_img)
    template_cv = load_image(template_img)

    # Detect source faces
    source_faces = app.get(source_cv)
    if len(source_faces) == 0:
        return None, None, None, "Không detect face trong source!"

    primary_source = source_faces[0]

    # Detect template faces (multi-support)
    template_faces = app.get(template_cv)
    if len(template_faces) == 0:
        return None, None, None, "Không detect face trong template!"

    # Handle multi-sources
    if multi_sources:
        multi_embeds = []
        for ms_img in multi_sources:
            ms_cv = load_image(ms_img)
            ms_faces = app.get(ms_cv)
            if ms_faces:
                multi_embeds.append(ms_faces[0].embedding)
        if len(multi_embeds) != len(template_faces):
            print(f"Warning: {len(multi_embeds)} sources for {len(template_faces)} template faces. Duplicating.")
            multi_embeds = (multi_embeds * (len(template_faces) // len(multi_embeds) + 1))[:len(template_faces)]
    else:
        multi_embeds = [primary_source.embedding] * len(template_faces)

    # Swap (dùng InSwapper: truyền Face source và target)
    result_cv = template_cv.copy()
    for t_face in template_faces:
        result_cv = inswapper.get(result_cv, t_face, primary_source, paste_back=True)

    # Enhance
    _, _, enhanced_cv = restorer.enhance(result_cv, has_aligned=False, only_center_face=False, paste_back=True)

    # Convert sang PIL
    enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))
    
    # Save output
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/swapped.png"
    enhanced_pil.save(output_path)

    return enhanced_pil, source_img, template_img, f"Swap thành công! {len(template_faces)} faces processed. Output saved: {output_path}"

# Gradio UI
with gr.Blocks(title="Face Swap Demo - Source + Template") as demo:
    gr.Markdown("# Face Swap Demo với InsightFace + Gradio")
    gr.Markdown("Upload source (single/multi via folder), template, click Swap. Output: Preview + Download.")

    with gr.Row():
        with gr.Column(scale=1):
            source_input = gr.Image(type="pil", label="Source Face (Single Image)")
            multi_source_input = gr.File(label="Multi-Sources (Folder ZIP - Optional)", file_types=[".zip"])
            template_input = gr.Image(type="pil", label="Template (Single/Multi-Face Image)")
            swap_btn = gr.Button("Swap Faces", variant="primary")
        
        with gr.Column(scale=2):
            with gr.Tab("Swapped Output"):
                output_img = gr.Image(label="Swapped Image")
            with gr.Tab("Source Preview"):
                source_preview = gr.Image(label="Source Image")
            with gr.Tab("Template Preview"):
                template_preview = gr.Image(label="Template Image")
            status = gr.Textbox(label="Status")

    swap_btn.click(
        fn=swap_faces,
        inputs=[source_input, template_input, multi_source_input],
        outputs=[output_img, source_preview, template_preview, status]
    )

    gr.Examples(
        examples=[
            ["samples/source.jpg", "samples/magic_output1.png"],
            ["samples/source.jpg", "samples/mermaid_output1.webp"]
        ],
        inputs=[source_input, template_input],
        fn=swap_faces,
        outputs=[output_img, source_preview, template_preview, status]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)