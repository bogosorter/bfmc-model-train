import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train26/weights/best.pt")

results = model.predict(source="input.png")

img = results[0].plot()  # annotated BGR image
cv2.imshow("Predictions", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
