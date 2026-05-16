from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.train(data="datasets/bfmc-generated/data.yaml", epochs=15, imgsz=416)

model.predict(source="test")
