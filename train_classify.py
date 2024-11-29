import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from collections import Counter
import os

# Charger les données
data_path = 'data.pickle'
if not os.path.exists(data_path):
    print(f"Erreur : Le fichier '{data_path}' est introuvable.")
    exit(1)

with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)

# Extraire les données et les étiquettes
data = data_dict['data']
labels = data_dict['labels']

# Vérification et filtrage des données
correct_length = 42  # Assurez-vous que toutes les entrées ont 42 caractéristiques
filtered_data = []
filtered_labels = []

for i in range(len(data)):
    if len(data[i]) == correct_length:
        filtered_data.append(data[i])
        filtered_labels.append(labels[i])

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Filtrer les classes avec au moins 2 exemples
label_counts = Counter(labels)
valid_labels = [label for label, count in label_counts.items() if count >= 2]

filtered_data = []
filtered_labels = []
for i in range(len(labels)):
    if labels[i] in valid_labels:
        filtered_data.append(data[i])
        filtered_labels.append(labels[i])

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Afficher des informations sur les données filtrées
print(f"Nombre de données après filtrage : {data.shape[0]}")
print(f"Répartition des classes : {Counter(labels)}")

# Diviser les données en ensemble d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Modèle de classification
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
model.fit(x_train, y_train)

# Prédire les résultats
y_predict = model.predict(x_test)

# Calculer l'accuracy
score = accuracy_score(y_test, y_predict)

print(f"{score * 100:.2f}% des échantillons ont été classés correctement !")

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_predict))

# Matrice de confusion
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_predict))

# Sauvegarder le modèle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Le modèle a été sauvegardé dans 'model.p'.")
