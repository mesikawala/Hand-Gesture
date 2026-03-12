# main.py
import cv2
from move import HandSystem
from fps_counter import FPSCounter

cap = cv2.VideoCapture(0)
system = HandSystem()
fps_counter = FPSCounter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame = system.result_hand(frame)
    frame = system.render_sphere(frame)

    fps = fps_counter.update()

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (frame.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Detection - Tasks API", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()