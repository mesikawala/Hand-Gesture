# move.py
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
base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

hand_detector = vision.HandLandmarker.create_from_options(options)


# =========================================================
# Hand Connections (for drawing skeleton)
# =========================================================

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


# =========================================================
# Gesture Detection Helpers
# =========================================================

def is_fist(hand_landmarks):
    tips = [8, 12, 16, 20]

    for tip in tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            return False

    return True


# =========================================================
# Sphere Generator
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

# =========================================================
# 3D → 2D Projection
# =========================================================

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

        # Sphere state
        self.sphere_visible = False
        self.previous_gesture = None

        # Rotation angles
        self.angle_x = 0
        self.angle_y = 0

        # Explosion state
        self.prev_two_fist = False
        self.explode = False
        self.particles = []


    def hand_tracking(self, hand_landmarks):

        current_x = hand_landmarks[9].x
        current_y = hand_landmarks[9].y

        # =====================
        # Swipe rotation (Fist)
        # =====================

        if is_fist(hand_landmarks):
            if self.prev_x is not None:
                dx = current_x - self.prev_x
                dy = current_y - self.prev_y

                self.angle_y += dx * 5
                self.angle_x += dy * 5

            self.prev_x = current_x
            self.prev_y = current_y

        else:

            self.prev_x = None
            self.prev_y = None


        # =====================
        # Peace gesture detect
        # =====================

        index_up  = hand_landmarks[8].y  < hand_landmarks[6].y
        middle_up = hand_landmarks[12].y < hand_landmarks[10].y

        ring_down  = hand_landmarks[16].y > hand_landmarks[14].y
        pinky_down = hand_landmarks[20].y > hand_landmarks[18].y

        if index_up and middle_up and ring_down and pinky_down:
            return "PEACE"

        return None


    def result_hand(self, image):

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        result = hand_detector.detect(mp_image)

        if result.hand_landmarks:

            h, w, _ = image.shape

            for hand_landmarks in result.hand_landmarks:

                # =====================
                # Draw connections
                # =====================

                for start_idx, end_idx in HAND_CONNECTIONS:

                    x1 = int(hand_landmarks[start_idx].x * w)
                    y1 = int(hand_landmarks[start_idx].y * h)

                    x2 = int(hand_landmarks[end_idx].x * w)
                    y2 = int(hand_landmarks[end_idx].y * h)

                    cv2.line(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (255,255,255),
                        2
                    )


                # =====================
                # Draw landmarks
                # =====================

                for landmark in hand_landmarks:

                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)

                    cv2.circle(
                        image,
                        (cx, cy),
                        5,
                        (255,255,0),
                        -1
                    )


                # =====================
                # Gesture logic
                # =====================

                gesture = self.hand_tracking(hand_landmarks)

                if gesture == "PEACE" and self.previous_gesture != "PEACE":
                    self.sphere_visible = not self.sphere_visible

                self.previous_gesture = gesture


                # =====================
                # Explosion gesture
                # =====================

                fist_count = sum(
                    is_fist(h)
                    for h in result.hand_landmarks
                )

                two_fist_now = fist_count == 2


                if (
                    self.prev_two_fist
                    and not two_fist_now
                    and self.sphere_visible
                ):
                    self.explode = True
                    self.init_explosion()


                if (
                    not self.prev_two_fist
                    and two_fist_now
                ):
                    self.explode = False
                    self.particles = []


                self.prev_two_fist = two_fist_now

        return image


    def render_sphere(self, frame):

        if not self.sphere_visible:
            return frame

        h, w, _ = frame.shape


        # =====================
        # Explosion render
        # =====================

        if self.explode:

            self.update_particles()

            for p in self.particles:

                x2d, y2d = project_point(
                    p["pos"][0],
                    p["pos"][1],
                    p["pos"][2],
                    w,
                    h
                )

                cv2.circle(
                    frame,
                    (x2d, y2d),
                    2,
                    (0,255,255),
                    -1
                )

            return frame


        # =====================
        # Sphere rotation render
        # =====================

        cos_x = math.cos(self.angle_x)
        sin_x = math.sin(self.angle_x)

        cos_y = math.cos(self.angle_y)
        sin_y = math.sin(self.angle_y)


        for p in sphere_particles:

            # Rotate X
            y_rot = p[1] * cos_x - p[2] * sin_x
            z_rot = p[1] * sin_x + p[2] * cos_x

            # Rotate Y
            x_rot  = p[0] * cos_y + z_rot * sin_y
            z_final = -p[0] * sin_y + z_rot * cos_y

            x2d, y2d = project_point(
                x_rot,
                y_rot,
                z_final,
                w,
                h
            )

            cv2.circle(
                frame,
                (x2d, y2d),
                2,
                (0,255,255),
                -1
            )

        return frame


    def init_explosion(self):

        self.particles = []

        for p in sphere_particles:

            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            vz = random.uniform(-2, 2)

            self.particles.append({

                "pos": p.copy(),

                "vel": [
                    vx,
                    vy,
                    vz
                ]

            })


    def update_particles(self):

        for p in self.particles:

            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["pos"][2] += p["vel"][2]