import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import random

# =========================================================
# MediaPipe Hand Landmarker Init
# =========================================================
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20), (0,17)
]

# =========================================================
# Sphere Generator & Projection
# =========================================================
def generate_sphere(radius, detail):
    particles = []
    for i in range(detail):
        theta = 2 * math.pi * i / detail
        for j in range(detail):
            phi = math.pi * j / detail
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            particles.append([x, y, z])
    return particles

sphere_particles = generate_sphere(5, 20)

def project_point(x, y, z, w, h, scale=200, distance=10):
    factor = scale / (z + distance)
    x2d = int(x * factor + w / 2)
    y2d = int(y * factor + h / 2)
    return x2d, y2d

# =========================================================
# Main Hand System
# =========================================================
class HandSystem:
    def __init__(self):
        # Rotation tracking
        self.prev_x = None
        self.prev_y = None
        self.angle_x = 0
        self.angle_y = 0

        # State management
        self.sphere_visible = False
        self.explode = False
        self.reassembling = False
        self.particles = []
        self.last_landmarks = None

    def update_rotation(self, hand_landmarks):
        current_x = hand_landmarks[9].x
        current_y = hand_landmarks[9].y

        if self.prev_x is not None:
            dx = current_x - self.prev_x
            dy = current_y - self.prev_y
            
            if abs(dx) > 0.005:
                self.angle_y += dx * 7
            if abs(dy) > 0.01:
                self.angle_x += dy * 7
                

        self.prev_x = current_x
        self.prev_y = current_y

    def result_hand(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = hand_detector.detect(mp_image)

        if result.hand_landmarks:
            h, w, _ = image.shape
            self.last_landmarks = result.hand_landmarks[0]
            
            for hand_landmarks in result.hand_landmarks:
                # Gambar Skeleton
                for start_idx, end_idx in HAND_CONNECTIONS:
                    x1, y1 = int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h)
                    x2, y2 = int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h)
                    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Gambar Titik Landmark
                for landmark in hand_landmarks:
                    cv2.circle(image, (int(landmark.x * w), int(landmark.y * h)), 5, (255, 255, 0), -1)

        else:
            self.last_landmarks = None
            self.prev_x = None
            self.prev_y = None

        return image

    def render_sphere(self, frame):
        if not self.sphere_visible and not self.explode:
            return frame

        h, w, _ = frame.shape

        if self.explode:
            self.update_particles()
            for p in self.particles:
                x2d, y2d = project_point(p["pos"][0], p["pos"][1], p["pos"][2], w, h)
                cv2.circle(frame, (x2d, y2d), 2, (0, 255, 255), -1)
            return frame

        # Sphere rotation render
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)
        cos_y, sin_y = math.cos(self.angle_y), math.sin(self.angle_y)

        for p in sphere_particles:
            y_rot = p[1] * cos_x - p[2] * sin_x
            z_rot = p[1] * sin_x + p[2] * cos_x
            x_rot = p[0] * cos_y + z_rot * sin_y
            z_final = -p[0] * sin_y + z_rot * cos_y
            x2d, y2d = project_point(x_rot, y_rot, z_final, w, h)
            cv2.circle(frame, (x2d, y2d), 2, (0, 255, 255), -1)
        return frame

    def init_explosion(self):
        self.particles = []
        for p in sphere_particles:
            self.particles.append({
                "pos": p.copy(),
                "vel": [random.uniform(-1.5, 1.5) for _ in range(3)]
            })

    def update_particles(self):
        all_arrived = True
        for i, p in enumerate(self.particles):
            if self.reassembling:
                target = sphere_particles[i]
                for j in range(3):
                    diff = target[j] - p["pos"][j]
                    if abs(diff) > 0.2:
                        p["pos"][j] += diff * 0.15
                        all_arrived = False
                    else:
                        p["pos"][j] = target[j]
            else:
                for j in range(3):
                    p["pos"][j] += p["vel"][j]
                all_arrived = False

        if self.reassembling and all_arrived:
            self.explode = False
            self.reassembling = False
            self.sphere_visible = True