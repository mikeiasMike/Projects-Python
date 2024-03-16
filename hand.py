import cv2
import mediapipe as mp

hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

while True:
    # Lendo um quadro do vídeo
    ret, frame = cap.read()

    # Convertendo a imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Gerando resultados do MediaPipe
    results = hands.process(image)

    # Obtendo a posição da mão
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

        # Movendo o cursor
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.moveWindow("Controle do Cursor", x, y)

    # Mostrando a imagem
    cv2.imshow("Controle do Cursor", frame)

    # Aperte 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
