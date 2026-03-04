import torch
import numpy as np
# --- LE FIX DÉFINITIF POUR PYTORCH 2.6+ ---
# On force weights_only à False globalement pour ce script
import torch.serialization
original_load = torch.load

def patched_load(*args, **kwargs):
    # On force l'argument weights_only à False
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load
# ------------------------------------------

import cv2
from hsemotion.facial_emotions import HSEmotionRecognizer

class EmotionRegressor:
    def __init__(self):
        # Cette fois, ça passera sans erreur car torch.load est "patché"
        self.model = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device='cpu')

    def get_va(self, face_img_rgb):
        # Conversion pour hsemotion
        face_img_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
        
        # Récupération des scores bruts (logits)
        _, va = self.model.predict_emotions(face_img_bgr, logits=True)
        v_raw, a_raw = va[0], va[1]
        
        # --- CORRECTION DES SIGNES ---
        # Si sourire = Valence négative -> on inverse
        v_corrected = -v_raw 
        # Si sourire dents (intense) = Arousal négatif -> on inverse
        a_corrected = -a_raw 

        # --- SQUASHING (Normalisation -1 à 1) ---
        # Utilisation de la tangente hyperbolique pour écraser les valeurs extrêmes
        valence = np.tanh(v_corrected)
        arousal = np.tanh(a_corrected)
        
        return valence, arousal