import os

import numpy as np
from hailo_sdk_client import ClientRunner
from PIL import Image

# --- Config ---
ONNX_PATH = "/local/shared_with_docker/best.onnx"
CALIB_PATH = "/local/shared_with_docker/images"
HW_ARCH = "hailo8"
MODEL_NAME = "yolov11s"
INPUT_SIZE = (416, 416)

# --- Load calibration images ---
calib_dataset = np.array(
    [
        np.array(
            Image.open(os.path.join(CALIB_PATH, f)).convert("RGB").resize(INPUT_SIZE),
            dtype=np.float32,
        )
        for f in os.listdir(CALIB_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

print(f"Calibration dataset shape: {calib_dataset.shape}")
print(f"Value range: [{calib_dataset.min():.3f}, {calib_dataset.max():.3f}]")

# --- Parse ---
runner = ClientRunner(hw_arch=HW_ARCH)

hn, npz = runner.translate_onnx_model(
    ONNX_PATH,
    MODEL_NAME,
    end_node_names=[
        "/model.23/cv2.0/cv2.0.2/Conv",  # reg stride 8  -> conv51
        "/model.23/cv3.0/cv3.0.2/Conv",  # cls stride 8  -> conv54
        "/model.23/cv2.1/cv2.1.2/Conv",  # reg stride 16 -> conv62
        "/model.23/cv3.1/cv3.1.2/Conv",  # cls stride 16 -> conv65
        "/model.23/cv2.2/cv2.2.2/Conv",  # reg stride 32 -> conv77
        "/model.23/cv3.2/cv3.2.2/Conv",  # cls stride 32 -> conv80
    ],
)

runner.save_har(f"{MODEL_NAME}_parsed.har")
print("Saved parsed model.")

# --- Optimize ---
runner.load_model_script("""
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
quantization_param([conv54], force_range_out=[0.0, 1.0])
quantization_param([conv65], force_range_out=[0.0, 1.0])
quantization_param([conv80], force_range_out=[0.0, 1.0])
nms_postprocess("yolov11s_nms_config.json", meta_arch=yolov8, engine=cpu)
""")

runner.optimize(calib_dataset)
runner.save_har(f"{MODEL_NAME}_quantized.har")
print("Saved quantized model.")

# --- Compile ---
compiled_hef = runner.compile()

with open(f"{MODEL_NAME}.hef", "wb") as f:
    f.write(compiled_hef)

print(f"Done — saved {MODEL_NAME}.hef")
