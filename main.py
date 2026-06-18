from ultralytics import YOLO

model = YOLO("runs/detect/train41/weights/last.pt")
results = model.train(data="datasets/bfmc-generated/data.yaml", epochs=4, imgsz=416)

model.predict(source="test")
