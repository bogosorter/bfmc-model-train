from ultralytics import YOLO

model = YOLO("runs/detect/train33/weights/best.pt")

# Export to HEF (Hailo Executable Format)
model.export(format="onnx", imgsz=416, opset=17)
