# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import time
from collections import deque

# --- CONFIGURATION DU MODÈLE ---
MODEL_PATH = "/home/elouand/.hsemotion/enet_b0_8_va_mtl.pt"

class VADModel(nn.Module):
    def __init__(self):
        super(VADModel, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=10)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        if hasattr(checkpoint, 'state_dict'):
            self.base_model.load_state_dict(checkpoint.state_dict())
        else:
            self.base_model.load_state_dict(checkpoint)
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)

# --- PROXY PEPPER (TA CONFIGURATION) ---
def apply_pepper_proxy(frame):
    # Simulation résolution Pepper
    frame = cv2.resize(frame, (640, 480))
    # Bruit numérique (Grain 30)
    noise = np.random.randint(0, 30, (480, 640, 3), dtype='uint8')
    frame = cv2.add(frame, noise)
    # Compression JPEG (Qualité 30)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, encimg = cv2.imencode('.jpg', frame, encode_param)
    return cv2.imdecode(encimg, 1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VADModel().to(device)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Lissage temporel pour stabiliser l'Arousal (Moyenne glissante sur 10 frames)
    history_v = deque(maxlen=10)
    history_a = deque(maxlen=10)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    print("--- Pepper-Vision VAD Proxy Actif ---")

    while True:
        ret, raw_frame = cap.read()
        if not ret: break
        
        # 1. APPLICATION DU PROXY
        frame = apply_pepper_proxy(raw_frame)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))

        v_final, a_final, d_final = 0.0, 0.0, 0.0

        for (x, y, w, h) in faces:
            # ROI : On découpe le visage du carré vert
            face_img = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            img_tensor = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                scores = output[0].cpu().numpy()

            # Extraction VAD (MTL : 8 émotions + Valence[8] + Arousal[9])
            v_raw = float(scores[8])
            a_raw = float(scores[9])

            # 2. LISSAGE (Règle le problème de l'Arousal qui ne descend pas)
            history_v.append(v_raw)
            history_a.append(a_raw)
            
            v_final = sum(history_v) / len(history_v)
            a_final = sum(history_a) / len(history_a)
            
            # 3. DOMINANCE (Calculée par l'intensité de l'état émotionnel)
            # Plus l'utilisateur est expressif (V) et énergique (A), plus il domine.
            d_final = (a_final + abs(v_final)) / 2

            # Affichage
            color = (0, 255, 0) if v_final >= 0 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            break

        # Dashboard UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"VALENCE:   {v_final:+.2f}", (25, 45), 1, 1.4, (255, 255, 255), 2)
        cv2.putText(frame, f"AROUSAL:   {a_final:+.2f}", (25, 85), 1, 1.4, (255, 255, 255), 2)
        cv2.putText(frame, f"DOMINANCE: {d_final:+.2f}", (25, 125), 1, 1.4, (255, 255, 255), 2)

        cv2.imshow('Pepper-Vision VAD Manual', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()