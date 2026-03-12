import cv2
import numpy as np
import os
import sys

# On s'assure que le dossier est dans le chemin Python pour l'import
sys.path.append(os.path.join(os.path.dirname(__file__), 'reconnaissanceVideo'))

from traitementVideo import EmotionRegressor

# Singleton pour le modèle
_regressor_instance = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_visual_vad(frame):
    """Fonction appelée par tes autres scripts (receveur, etc.)"""
    global _regressor_instance
    
    if _regressor_instance is None:
        print("Chargement de l'EmotionRegressor...")
        _regressor_instance = EmotionRegressor()

    v, a, d = 0.0, 0.0, 0.0

    # Détection de visage robuste (remplace MediaPipe)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Extraction et conversion
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Appel de la logique du fichier traitementVideo
        scores = _regressor_instance.get_vad(face_rgb)
        v, a, d = scores[0], scores[1], scores[2]
        break 

    return np.array([v, a, d])