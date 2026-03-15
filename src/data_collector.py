import cv2
import csv
import os
import numpy as np
from src.move import HandSystem

# ---- Inisialisasi ----
cap = cv2.VideoCapture(0)
system = HandSystem()
file_name = 'dataset_isyarat.csv'

# siapkan CSV
if not os.path.exists(file_name):
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f'c{i}' for i in range(63)] + ['label']
        writer.writerow(header)

print("--- MODE PENGUMPULAN DATA ISYARAT ---")
print("Panduan Label:")
print("0: FIST (Kepal) | 1: PEACE | 2: HURUF L | 3: HURUF O | 4: IDLE (Rileks)")
print("-------------------------------------")
print("CARA REKAM: TAHAN angka pada keyboard sambil gerakkan tangan.")
print("Tekan 'q' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame = system.result_hand(frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Logika simpan data
    label = None

    if ord('0') <= key <= ord('4'):
        label = int(chr(key))

    if label is not None:
        if hasattr(system, 'last_landmarks') and system.last_landmarks:
            row = []
            for lm in system.last_landmarks:
                row.extend([lm.x, lm.y, lm.z])
            row.append(label)

            with open(file_name, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Berhasil simpan data label {label}")

            # Notifikasi visual di layar saat merekam
            cv2.putText(frame, f"RECORDING LABEL: {label}", (20, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) # Lampu merah tanda rekam

        else:
            print("Tangan tidak terdeteksi! Pastikan tangan ada di frame")
        # Instruksi di layar
        cv2.putText(frame, "Hold 0-4 to Record | 'q' to Quit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Data Collector AI", frame)

cap.release()
cv2.destroyAllWindows()