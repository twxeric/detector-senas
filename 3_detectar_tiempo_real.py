import cv2
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque, Counter

MODEL_PATH = "hand_landmarker.task"

with open("modelo_letras.pkl", "rb") as f:
    modelo = pickle.load(f)

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
historial = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer la camara")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    letra = "Sin mano"
    confianza_txt = ""

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        x0, y0, z0 = hand[0].x, hand[0].y, hand[0].z
        fila = []
        for lm in hand:
            fila.extend([lm.x - x0, lm.y - y0, lm.z - z0])

        pred = modelo.predict([fila])[0]
        if hasattr(modelo, "predict_proba"):
            conf = max(modelo.predict_proba([fila])[0])
            confianza_txt = f"{conf:.2f}"

        historial.append(pred)
        letra = Counter(historial).most_common(1)[0][0]

        for lm in hand:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.rectangle(frame, (10, 10), (380, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Letra: {letra}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"Confianza: {confianza_txt}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("Detector de letras", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()