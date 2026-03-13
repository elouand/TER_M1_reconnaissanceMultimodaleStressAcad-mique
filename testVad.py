import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import time
 
# --- CONFIGURATION DU MODÈLE ---
MODEL_PATH = "./.hsemotion/enet_b0_8_va_mtl.pt"  # Vérifie ce chemin

import os

# Helper pour valider la présence du fichier de modèle
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier de modèle n'a pas été trouvé : {MODEL_PATH}.\n" \
                             "Vérifiez que le chemin est correct et que le modèle a été téléchargé.")

class VADModel(nn.Module):
    """ On encapsule le modèle pour un accès direct aux sorties """
    def __init__(self):
        super(VADModel, self).__init__()
        # Création de l'architecture EfficientNet-B0
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=10)
        # Chargement des poids manuellement
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        if hasattr(checkpoint, 'state_dict'):
            self.base_model.load_state_dict(checkpoint.state_dict())
        else:
            self.base_model.load_state_dict(checkpoint)
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)

def main():
    # 1. Chargement du modèle et des transformations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VADModel().to(device)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Détecteur de visage OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détection
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # ROI : On découpe le visage du carré vert
            face_img = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Passage en Tenseur
            img_tensor = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(device)

            # Inférence directe
            with torch.no_grad():
                output = model(img_tensor)
                scores = output[0].cpu().numpy()

            # Extraction VAD (Indices spécifiques au modèle MTL)
            # Valence = index 8 | Arousal = index 9
            v = float(scores[8])
            a = float(scores[9])
            # Calcul de la Dominance : D = (Arousal + |Valence|) / 2
            d = (a + abs(v)) / 2

            # Affichage
            color = (0, 255, 0) if v >= 0 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"V: {v:.2f} A: {a:.2f} D: {d:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            break

        cv2.imshow('Pepper-Vision VAD Manual', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()