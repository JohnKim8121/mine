# Soccer Offside Detection

This project contains a simple Flask application that demonstrates offside detection using a webcam feed. It relies on a pre-trained YOLOv5 model to detect players and the ball from each frame.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Running

Start the Flask server:

```bash
python -m offside_app.app
```

Open `http://localhost:5000` in your browser to view the live video feed. The application will attempt to highlight players and the ball and display `OFFSIDE!` on the screen when the naive offside check triggers.

This is a basic proof of concept and does **not** implement full rules of soccer offside detection.
