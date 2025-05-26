import cv2
import mediapipe as mp
import time

# Inicializa MediaPipe
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic  # Holistic = cuerpo + manos + cara
mp_drawing = mp.solutions.drawing_utils

# Guarda los movimientos en una lista
movimientos = []

# Abre la cámara
cap = cv2.VideoCapture(0)

# Verifica si la cámara funciona
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Procesamiento con Holistic
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Dibuja cuerpo completo
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

        # Dibuja cara
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1))

        # Dibuja manos
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Registra los movimientos: nariz, ojos, labios, manos
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nariz = landmarks[mp_pose.PoseLandmark.NOSE]
            movimientos.append({
                "tiempo": time.time(),
                "nariz": (nariz.x, nariz.y, nariz.z)
            })

        cv2.imshow("Seguimiento Humano y Estructura Ósea", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cierra todo
cap.release()
cv2.destroyAllWindows()

# (Opcional) Guarda los datos
with open("movimientos.txt", "w") as f:
    for m in movimientos:
        f.write(str(m) + "\n")
