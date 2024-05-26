import numpy as np
import torch
import cv2

model = torch.hub.load("ultralytics/yolov5", "custom", path="models/best.pt")

camera = cv2.VideoCapture(0)

while camera.isOpened():

    ret, frame = camera.read()

    results = model(frame)

    cv2.imshow("YOLO", np.squeeze(results.render()))

    # If we press the exit-buttom 'q' we end the webcam caption
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
