import os
import zipfile
from glob import glob
from huggingface_hub import hf_hub_download

def ensure_models(model_dir="models"):
    """
    Kiểm tra models trong model_dir, chỉ tải nếu thiếu.
    Returns: Path đến model_dir chứa buffalo_l/ và inswapper_128.onnx.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Kiểm tra inswapper_128.onnx
    inswapper_path = os.path.join(model_dir, "inswapper_128.onnx")
    if not os.path.exists(inswapper_path):
        print("Tải inswapper_128.onnx từ Hugging Face...")
        hf_hub_download(
            repo_id="ezioruan/inswapper_128.onnx",
            filename="inswapper_128.onnx",
            local_dir=model_dir
        )
    else:
        print("inswapper_128.onnx đã tồn tại, skip tải.")
    
    # Kiểm tra buffalo_l folder
    buffalo_dir = os.path.join(model_dir, "buffalo_l")
    required_files = ["det_10g.onnx", "2d106det.onnx", "w600k_r50.onnx"]
    all_files_exist = all(os.path.exists(os.path.join(buffalo_dir, f)) for f in required_files)
    
    if all_files_exist:
        print("Tất cả buffalo_l files (det_10g.onnx, 2d106det.onnx, w600k_r50.onnx) đã tồn tại, skip tải.")
    else:
        # Thử tìm và giải nén zip nội bộ nếu có
        local_zip_candidates = [
            os.path.join(model_dir, "buffalo_l.zip"),
            *glob(os.path.join(model_dir, "buffalo*.zip"))
        ]
        local_zip_path = next((p for p in local_zip_candidates if os.path.exists(p)), None)
        if local_zip_path:
            print(f"Phát hiện zip cục bộ: {local_zip_path}. Đang giải nén vào {buffalo_dir}...")
            os.makedirs(buffalo_dir, exist_ok=True)
            with zipfile.ZipFile(local_zip_path, 'r') as zf:
                zf.extractall(buffalo_dir)
            # Kiểm tra lại sau khi giải nén
            all_files_exist = all(os.path.exists(os.path.join(buffalo_dir, f)) for f in required_files)
            if all_files_exist:
                print("Giải nén buffalo_l.zip thành công. Bỏ qua tải xuống.")
            else:
                print("Giải nén zip xong nhưng thiếu file cần thiết. Vui lòng kiểm tra nội dung zip.")
        else:
            print("Thiếu files buffalo_l, cần giải nén buffalo_l.zip từ https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
    
    return model_dir