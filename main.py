import cv2
import pickle
from src.move import HandSystem
from src.fps_counter import FPSCounter

with open('./models/model_gesture.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
system = HandSystem()
fps_counter = FPSCounter()
labels = {0: "FIST", 1: "PEACE", 4: "IDLE"} 
prev_nama_gesture = "IDLE"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = system.result_hand(frame)

    # =========================================================
    # LOGIKA PREDIKSI AI (REAL-TIME)
    # =========================================================
    if system.last_landmarks:
        coords = []
        for lm in system.last_landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        
        prediction = model.predict([coords])[0]
        nama_gesture = labels.get(prediction, "UNKNOWN")

        cv2.putText(frame, f"Gesture: {nama_gesture}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Kontrol Bola Berdasarkan Prediksi AI
        if nama_gesture == "PEACE":
            if prev_nama_gesture != "PEACE":
                system.sphere_visible = not system.sphere_visible
                system.explode = False
                print(f"Sphere Status: {system.sphere_visible}")

        elif nama_gesture == "FIST":
            if system.sphere_visible and not system.explode:
                system.update_rotation(system.last_landmarks)
                cv2.putText(frame, "STATUS: GRABBING", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        if nama_gesture != "FIST":
            system.prev_x = None
            system.prev_y = None

        prev_nama_gesture = nama_gesture
        
    else:
        prev_nama_gesture = "IDLE"
        system.prev_x = None
        system.prev_y = None

    # Gambar bola/partikel
    frame = system.render_sphere(frame)

    # Update FPS
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

    cv2.imshow("Hand Detection AI - Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()