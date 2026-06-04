import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = vision.HandLandmarker.create_from_options(options)

def distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def dedo_extendido(hand, tip, pip):
    return hand[tip].y < hand[pip].y

def clasificar_letra(hand):
    thumb_tip = hand[4]
    index_tip = hand[8]
    index_pip = hand[6]
    middle_tip = hand[12]
    middle_pip = hand[10]
    ring_tip = hand[16]
    ring_pip = hand[14]
    pinky_tip = hand[20]
    pinky_pip = hand[18]

    index_up = dedo_extendido(hand, 8, 6)
    middle_up = dedo_extendido(hand, 12, 10)
    ring_up = dedo_extendido(hand, 16, 14)
    pinky_up = dedo_extendido(hand, 20, 18)

    # Letra B: cuatro dedos arriba, pulgar recogido
    if index_up and middle_up and ring_up and pinky_up:
        if thumb_tip.x < hand[2].x or distancia(thumb_tip, hand[9]) < 0.12:
            return "B"

    # Letra V: índice y medio arriba, anular y meñique abajo
    if index_up and middle_up and not ring_up and not pinky_up:
        return "V"

    # Letra L: índice arriba y pulgar separado
    if index_up and not middle_up and not ring_up and not pinky_up:
        if distancia(thumb_tip, index_tip) > 0.18:
            return "L"

    # Letra A: puño cerrado con pulgar al costado
    if not index_up and not middle_up and not ring_up and not pinky_up:
        if distancia(thumb_tip, index_tip) > 0.08:
            return "A"

    return "Desconocida"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        for lm in hand_landmarks:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        letra = clasificar_letra(hand_landmarks)

    cv2.rectangle(frame, (10, 10), (320, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Letra: {letra}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Detector de letras", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 