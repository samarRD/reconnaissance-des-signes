import os
import cv2
import time

DATA_DIR = r'C:\Users\samar\OneDrive\Documents\projet (1)\projet'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
number_of_classes = 25  
dataset_size = 9 

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
else:
    for j in range(19, number_of_classes + 1):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'Préparez votre main pour la capture de la classe {j}.')
        time.sleep(2)  # Délai pour permettre à l'utilisateur de se préparer

        # Récupérer le nombre d'images déjà présentes dans le dossier pour cette classe
        existing_images = len(os.listdir(class_dir))
        counter = existing_images

        while counter < dataset_size + existing_images:  # Prendre en compte les images déjà présentes
            ret, frame = cap.read()
            if not ret:
                print("Erreur lors de la lecture de la caméra.")
                break
            cv2.putText(frame, 'Appuyez sur "C" pour capturer, "N" pour passer à la classe suivante', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Créer un nom unique pour chaque image en fonction du compteur
                cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
                print(f'Image {counter + 1} capturée pour la classe {j}.')
                counter += 1
            elif key == ord('n'):
                print(f'Passage à la classe suivante sans capturer d\'image.')
                break  # Passe à la classe suivante
            elif key == ord('q'):
                break  # Quitter l'application

    cap.release()
    cv2.destroyAllWindows()
