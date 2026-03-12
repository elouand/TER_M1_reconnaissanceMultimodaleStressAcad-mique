import torch
import numpy as np
import cv2
from hsemotion.facial_emotions import HSEmotionRecognizer

# Fix pour PyTorch 2.6+
import torch.serialization
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, weights_only=False, **kwargs)

class EmotionRegressor:
    def __init__(self, device='cpu'):
        # On utilise le modèle MTL EfficientNet-B0
        self.model = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device=device)

    def get_vad(self, face_img_rgb):
        """Calcule V, A et D selon TA logique de signes et de normalisation"""
        # HSEmotion attend du BGR pour son traitement interne
        face_img_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
        
        # Récupération des logits (scores bruts avant activation)
        _, va = self.model.predict_emotions(face_img_bgr, logits=True)
        v_raw, a_raw = va[0], va[1]
        
        # --- TA LOGIQUE DE SIGNES ---
        v_corrected = -v_raw 
        a_corrected = -a_raw 

        # --- TA LOGIQUE DE SQUASHING (tanh) ---
        valence = np.tanh(v_corrected)
        arousal = np.tanh(a_corrected)
        
        # --- CALCUL DE LA DOMINANCE ---
        # Basé sur ton code précédent : D = (A + |V|) / 2
        dominance = (arousal + abs(valence)) / 2
        
        return np.array([valence, arousal, dominance])