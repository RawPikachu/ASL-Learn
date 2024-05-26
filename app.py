from flask import Flask, render_template, Response
import numpy as np
import torch
import cv2
import time

camera = cv2.VideoCapture(0)
while not camera.isOpened():
    time.sleep(0.1)

app = Flask(__name__)

model = torch.hub.load("ultralytics/yolov5", "custom", path="models/best.pt")


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:
            results = model(frame)
            label, box = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
            ret, buffer = cv2.imencode(".jpg", np.squeeze(results.render()))
            frame = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )  # concat frame one by one and show result


@app.route("/video_feed")
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
