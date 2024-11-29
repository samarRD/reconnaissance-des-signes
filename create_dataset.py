import os
import cv2
import numpy as np
import pickle
import random
import mediapipe as mp

# Initialiser Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r'C:\Users\samar\OneDrive\Documents\projet (1)\projet'  # Chemin vers votre répertoire de données
correct_length = 42  # Nombre de points de repères à extraire (21 points * 2 coordonnées)

data = []
labels = []


# Fonction de prétraitement de l'image (redimensionnement, conversion en gris)
def preprocess_image(img):
    img = cv2.resize(img, (256, 256))  # Redimensionner l'image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)  # Appliquer un flou pour réduire le bruit
    return img_blur

# Fonction pour normaliser et extraire les landmarks
def extract_landmarks(img, results):
    data_aux = []
    x_ = []
    y_ = []
    
    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        # Normaliser les coordonnées
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))  # Normalisation par rapport à la coordonnée minimale x
            data_aux.append(y - min(y_))  # Normalisation par rapport à la coordonnée minimale y

    return data_aux, x_, y_

# Augmentation des données : rotation et ajustement de la luminosité
def augment_data(img):
    # Rotation aléatoire entre -15 et 15 degrés
    rows, cols = img.shape[:2]
    angle = random.randint(-15, 15)
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, matrix, (cols, rows))
    
    # Ajustement de la luminosité
    alpha = random.uniform(0.8, 1.2)  # Facteur de contraste
    beta = random.randint(-30, 30)  # Facteur de luminosité
    img_augmented = cv2.convertScaleAbs(img_rotated, alpha=alpha, beta=beta)

    return img_augmented

# Parcourir les données pour extraire les landmarks
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            img = cv2.imread(os.path.join(dir_path, img_path))
            
            # Prétraiter l'image (redimensionner, convertir en niveaux de gris)
            img_processed = preprocess_image(img)

            # Traiter l'image avec Mediapipe pour extraire les landmarks
            img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                data_aux, x_, y_ = extract_landmarks(img, results)

                # Vérifier que l'image contient les bonnes données (42 caractéristiques)
                if len(data_aux) == correct_length:
                    data.append(data_aux)
                    labels.append(dir_)

                    # Appliquer une augmentation des données (rotation et luminosité)
                    img_augmented = augment_data(img)
                    img_rgb = cv2.cvtColor(img_augmented, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)

                    if results.multi_hand_landmarks:
                        data_aux, x_, y_ = extract_landmarks(img_augmented, results)
                        if len(data_aux) == correct_length:
                            data.append(data_aux)
                            labels.append(dir_)

# Sauvegarder les données nettoyées et augmentées dans un fichier pickle
with open('cleaned_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
