import cv2

for i in range(5):
    print(f"Probando cámara índice {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Cámara funcional en índice {i}")
            cv2.imshow(f"Camara {i}", frame)
            cv2.waitKey(3000)  # muestra 3 segundos
            cv2.destroyAllWindows()
            cap.release()
            break
        else:
            print(f"La cámara {i} abre pero no entrega frame")
    else:
        print(f"No se pudo abrir la cámara {i}")

    cap.release()
else:
    print("No se encontró ninguna cámara funcional.")