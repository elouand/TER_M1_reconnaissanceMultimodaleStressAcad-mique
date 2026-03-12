import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "./.hsemotion/enet_b0_8_va_mtl.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialisation du détecteur de visage (chargé une seule fois)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Transformations d'image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VADModel(nn.Module):
    def __init__(self):
        super(VADModel, self).__init__()
        # Architecture EfficientNet-B0 pour 10 classes (MTL)
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=10)
        
        if os.path.isfile(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            if hasattr(checkpoint, 'state_dict'):
                self.base_model.load_state_dict(checkpoint.state_dict())
            else:
                self.base_model.load_state_dict(checkpoint)
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)

# Instance globale pour éviter de recharger le modèle à chaque frame
_model_instance = None

def get_visual_vad(frame):
    global _model_instance
    
    # Chargement paresseux du modèle
    if _model_instance is None:
        print(f"Chargement du modèle VAD sur {DEVICE}...")
        _model_instance = VADModel().to(DEVICE)

    # Valeurs par défaut (Neutre) si aucun visage n'est détecté
    v, a, d = 0.0, 0.0, 0.0

    # Prétraitement pour la détection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Extraction et conversion du visage
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Inférence
        img_tensor = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = _model_instance(img_tensor)
            scores = output[0].cpu().numpy()

        # Extraction selon les indices HSEmotion MTL
        v = float(scores[8]) # Valence
        a = float(scores[9]) # Arousal
        # Calcul de la Dominance : D = (Arousal + |Valence|) / 2
        d = (a + abs(v)) / 2
        
        # On ne traite que le premier visage détecté pour la performance
        break 

    return np.array([v, a, d])