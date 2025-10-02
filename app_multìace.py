import cv2
import numpy as np
from PIL import Image
import gradio as gr
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer
from models.model_downloader import ensure_models
import os
import zipfile
import tempfile

# Tải/check models
try:
    model_dir = ensure_models()
except Exception as e:
    raise Exception(f"Lỗi kiểm tra models: {str(e)}")

# Khởi tạo InsightFace
try:
    insight_root = os.path.abspath(os.path.join(model_dir, os.pardir))
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], root=insight_root)
    app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    raise Exception(f"Lỗi khởi tạo InsightFace: {str(e)}. Kiểm tra models/buffalo_l/ có det_10g.onnx, 2d106det.onnx, w600k_r50.onnx")

# InSwapper
try:
    inswapper = get_model(os.path.join(model_dir, 'inswapper_128.onnx'), providers=['CPUExecutionProvider'])
except Exception as e:
    raise Exception(f"Lỗi khởi tạo InSwapper: {str(e)}. Kiểm tra models/inswapper_128.onnx")

# GFPGAN
try:
    restorer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None)
except Exception as e:
    raise Exception(f"Lỗi khởi tạo GFPGAN: {str(e)}")

def load_image(image):
    """Load từ Gradio Image (PIL) hoặc file path sang OpenCV, resize nếu >512x512."""
    if image is None:
        return None
    try:
        # Nếu image là tuple (path, PIL_image) từ Gradio ZIP
        if isinstance(image, tuple):
            image = image[1] if len(image) > 1 else Image.open(image[0])
        # Nếu image là file path
        elif isinstance(image, str):
            image = Image.open(image)
        # Đảm bảo image là PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError(f"Input không phải PIL Image: {type(image)}")
        
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Resize nếu lớn hơn 512x512
        height, width = img_bgr.shape[:2]
        if max(width, height) > 512:
            scale = 512 / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return img_bgr
    except Exception as e:
        print(f"Lỗi load ảnh: {str(e)}")
        return None

def swap_faces(source_img, source_img_2, template_img, multi_sources=None):
    try:
        if template_img is None:
            return None, [], None, "Vui lòng upload template!"

        print("Loading template image...")
        template_cv = load_image(template_img)
        if template_cv is None:
            return None, [], None, "Lỗi load template image!"

        # Detect template faces
        print("Detecting template faces...")
        template_faces = app.get(template_cv)
        if len(template_faces) == 0:
            return None, [], None, "Không detect face trong template!"

        # Collect source faces
        source_faces = []
        source_previews = []

        # Source Face 1
        if source_img:
            print("Loading Source Face 1...")
            source_cv = load_image(source_img)
            if source_cv is None:
                return None, [], None, "Lỗi load Source Face 1!"
            faces = app.get(source_cv)
            if faces:
                source_faces.append(faces[0])
                source_previews.append(source_img)
            else:
                return None, [], None, "Không detect face trong Source Face 1!"

        # Source Face 2
        if source_img_2:
            print("Loading Source Face 2...")
            source_cv_2 = load_image(source_img_2)
            if source_cv_2 is None:
                return None, [], None, "Lỗi load Source Face 2!"
            faces_2 = app.get(source_cv_2)
            if faces_2:
                source_faces.append(faces_2[0])
                source_previews.append(source_img_2)
            else:
                return None, [], None, "Không detect face trong Source Face 2!"

        # Multi-sources (ZIP)
        if multi_sources:
            print("Processing multi-sources (ZIP)...")
            try:
                tmp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(multi_sources, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)

                for i, fname in enumerate(os.listdir(tmp_dir)):
                    file_path = os.path.join(tmp_dir, fname)
                    print(f"Loading source image {i+1} từ ZIP...")
                    ms_cv = load_image(file_path)
                    if ms_cv is None:
                        print(f"Warning: Không load được source image {i+1}")
                        continue
                    ms_faces = app.get(ms_cv)
                    if ms_faces:
                        source_faces.append(ms_faces[0])
                        source_previews.append(Image.open(file_path))
                    else:
                        print(f"Warning: Không detect face trong source image {i+1}")
                        continue
            except Exception as e:
                return None, [], None, f"Lỗi xử lý ZIP: {str(e)}"

        if not source_faces:
            return None, [], None, "Không detect face trong bất kỳ source image nào!"

        # Ánh xạ số lượng
        if len(source_faces) < len(template_faces):
            print(f"Warning: {len(source_faces)} source faces cho {len(template_faces)} template faces. Duplicating last source face.")
            source_faces = (source_faces * (len(template_faces) // len(source_faces) + 1))[:len(template_faces)]
            source_previews = (source_previews * (len(template_faces) // len(source_previews) + 1))[:len(template_faces)]
        elif len(source_faces) > len(template_faces):
            print(f"Warning: {len(source_faces)} source faces nhưng chỉ {len(template_faces)} template faces. Dùng {len(template_faces)} source faces đầu tiên.")
            source_faces = source_faces[:len(template_faces)]
            source_previews = source_previews[:len(template_faces)]

        # Swap
        print(f"Swapping {len(template_faces)} faces...")
        result_cv = template_cv.copy()
        for i, (t_face, s_face) in enumerate(zip(template_faces, source_faces)):
            print(f"Swapping face {i+1}...")
            result_cv = inswapper.get(result_cv, t_face, s_face, paste_back=True)

        # Enhance
        print("Enhancing with GFPGAN...")
        _, _, enhanced_cv = restorer.enhance(result_cv, has_aligned=False, only_center_face=False, paste_back=True)

        # Convert sang PIL
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))

        # Save output
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/swapped.png"
        enhanced_pil.save(output_path)

        return enhanced_pil, source_previews, template_img, f"Swap thành công! {len(template_faces)} faces processed. Output saved: {output_path}"
    
    except Exception as e:
        return None, [], None, f"Lỗi swap: {str(e)}"

# Gradio UI
with gr.Blocks(title="Face Swap Demo - Source + Template") as demo:
    gr.Markdown("# Face Swap Demo với InsightFace + Gradio")
    gr.Markdown("Upload ZIP (e.g., 2 faces: source1.jpg, source2.jpg) vào Multi-Sources, template (e.g., Mermaid with 2 faces), click Swap. Ảnh >512x512 tự resize.")

    with gr.Row():
        with gr.Column(scale=1):
            source_input = gr.Image(type="pil", label="Source Face 1 (e.g., Mặt bạn, optional)")
            source_input_2 = gr.Image(type="pil", label="Source Face 2 (e.g., Mặt bạn của bạn, optional)")
            multi_source_input = gr.File(label="Multi-Sources (ZIP chứa 2 faces, e.g., source1.jpg, source2.jpg)", file_types=[".zip"])
            template_input = gr.Image(type="pil", label="Template (e.g., Mermaid with 2 faces)")
            swap_btn = gr.Button("Swap Faces", variant="primary")
        
        with gr.Column(scale=2):
            with gr.Tab("Swapped Output"):
                output_img = gr.Image(label="Swapped Image")
            with gr.Tab("Source Previews"):
                source_preview = gr.Gallery(label="Source Images", elem_id="source-gallery")
            with gr.Tab("Template Preview"):
                template_preview = gr.Image(label="Template Image")
            status = gr.Textbox(label="Status")

    swap_btn.click(
        fn=swap_faces,
        inputs=[source_input, source_input_2, template_input, multi_source_input],
        outputs=[output_img, source_preview, template_preview, status]
    )

    # gr.Examples(
    #     examples=[
    #         [None, None, "samples/magic_output1.jpg", None],
    #         [None, None, "samples/mermaid_output1.webp", "samples/multi_sources.zip"],
    #     ],
    #     inputs=[source_input, source_input_2, template_input, multi_source_input],
    #     fn=swap_faces,
    #     outputs=[output_img, source_preview, template_preview, status]
    # )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861, debug=True)