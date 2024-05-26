from flask import Flask, render_template, Response, jsonify
import numpy as np
import torch
import cv2
import time

camera = cv2.VideoCapture(0)
while not camera.isOpened():
    time.sleep(0.1)

app = Flask(__name__, static_folder="assets", static_url_path="/assets")

model = torch.hub.load("ultralytics/yolov5", "custom", path="models/best.pt")

results = None
frame = None


def gen_frames():
    global results, frame

    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            results = model(frame)
            ret, buffer = cv2.imencode(".jpg", np.squeeze(results.render()))
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("start.html")


@app.route("/letters")
def letters():
    return render_template("letters.html")


@app.route("/words")
def words():
    return render_template("words.html")


@app.route("/realtime")
def realtime():
    return render_template("realtime.html")


@app.route("/letter_check")
def letter_check():
    return render_template("letter_check.html")


@app.route("/check_prediction")
def check_prediction():
    condition = False

    label, box = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    for i in range(len(label)):
        if label[i] == 0 and box[i][4] > 0.90:
            condition = True

    return jsonify(condition=condition)


@app.route("/fufilled")
def fufilled():
    global results, frame
    results = None
    frame = None
    return render_template("fufilled.html")


@app.route("/leaderboard")
def leaderboard():
    return render_template("leaderboard.html")


@app.route("/lettersB")
def lettersB():
    return render_template("lettersB.html")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
