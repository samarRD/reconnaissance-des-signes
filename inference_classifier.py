import pickle
import cv2
import mediapipe as mp
import numpy as np

# Charger le modèle enregistré
try:
    model_dict = pickle.load(open(r'C:\Users\samar\OneDrive\Documents\projet (1)\projet\model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Erreur : Le fichier du modèle 'model.p' est introuvable.")
    exit(1)
except Exception as e:
    print(f"Une erreur est survenue lors du chargement du modèle : {e}")
    exit(1)

# Initialisation de Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Dictionnaire des étiquettes pour les lettres
labels_dict = {
    0: 'A', 1: 'B', 2: 'L', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G',
    8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'M', 13: 'N', 14: 'O', 
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de la caméra.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection de la main
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extraire et normaliser les coordonnées
            for point in hand_landmarks.landmark:
                x, y = point.x, point.y
                x_.append(x)
                y_.append(y)

            min_x, min_y = min(x_), min(y_)
            for point in hand_landmarks.landmark:
                data_aux.append(point.x - min_x)
                data_aux.append(point.y - min_y)

        # Vérifier le nombre de caractéristiques
        if len(data_aux) == 42:
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), "Inconnu")

                # Déterminer les dimensions de la boîte englobante
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

                # Afficher la prédiction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except Exception as e:
                print(f"Erreur de prédiction : {e}")
        else:
            print("Erreur : Nombre incorrect de caractéristiques détectées.")

    # Afficher le flux vidéo
    cv2.imshow('Reconnaissance de gestes', frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
