from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.train(data="datasets/bfmc-generated/data.yaml", epochs=4, imgsz=416)

model.predict(source="test")
