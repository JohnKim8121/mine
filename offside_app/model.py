import cv2
import torch

class OffsideDetector:
    """Simple offside detection using YOLOv5."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(self.device)

    def detect(self, frame):
        """Detect players and ball in a frame."""
        results = self.model(frame)
        detections = results.xyxy[0]
        players = [d for d in detections if int(d[5]) == 0]  # class 0 = person
        ball = [d for d in detections if int(d[5]) == 32]   # class 32 = sports ball
        return players, ball

    def is_offside(self, players, ball, orientation: str = "horizontal") -> bool:
        """Naive offside check based on bounding box positions."""
        if not players or not ball:
            return False
        # pick player closest to opponent goal
        if orientation == "horizontal":
            player_max = max(players, key=lambda x: x[0])
            ball_x = ball[0][0]
            return player_max[0] > ball_x
        else:
            player_max = max(players, key=lambda x: x[1])
            ball_y = ball[0][1]
            return player_max[1] > ball_y
