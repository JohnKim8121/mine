from flask import Flask, Response, render_template
import cv2
from model import OffsideDetector

app = Flask(__name__)
detector = OffsideDetector()

camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        players, ball = detector.detect(frame)
        offside = detector.is_offside(players, ball)

        # draw boxes
        for p in players:
            x1, y1, x2, y2 = map(int, p[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for b in ball:
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if offside:
            cv2.putText(frame, "OFFSIDE!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
