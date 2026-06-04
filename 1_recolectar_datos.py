import cv2
import csv
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
OUTPUT_CSV = "datos_letras.csv"

LETRAS = ["A", "B", "L", "V"]
MUESTRAS_POR_LETRA = 200

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]
        writer.writerow(header)

for letra in LETRAS:
    print(f"\nPrepárate para la letra: {letra}")
    print("Presiona ESPACIO para comenzar a guardar muestras...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Letra: {letra} | ESPACIO = iniciar", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Recoleccion", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    contador = 0
    while contador < MUESTRAS_POR_LETRA:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            x0, y0, z0 = hand[0].x, hand[0].y, hand[0].z
            fila = []
            for lm in hand:
                fila.extend([lm.x - x0, lm.y - y0, lm.z - z0])

            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fila + [letra])

            contador += 1

            for lm in hand:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.putText(frame, f"Letra: {letra} | Muestras: {contador}/{MUESTRAS_POR_LETRA}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Recoleccion", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Datos guardados en {OUTPUT_CSV}")