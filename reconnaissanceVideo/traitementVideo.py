import torch
import numpy as np
import cv2
from hsemotion.facial_emotions import HSEmotionRecognizer
import os

# Fix pour PyTorch 2.6+
import torch.serialization

original_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # On n'ajoute weights_only que si l'appelant ne l'a pas déjà spécifié
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

# Application du patch sécurisé
torch.load = _patched_torch_load

class EmotionRegressor:
    def __init__(self, device='cpu'):
        # On définit le chemin relatif vers ton dossier .hsemotion dans le projet
        # 'os.path.dirname(__file__)' nous donne le dossier 'reconnaissanceVideo'
        # On remonte d'un cran pour atteindre le dossier racine du projet
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, ".hsemotion", "enet_b0_8_va_mtl.pt")
        
        if os.path.exists(model_path):
            print(f"Succès : Modèle local trouvé dans le projet -> {model_path}")
            # Note : On initialise avec le nom du modèle. 
            # Si le fichier est déjà dans ~/.hsemotion ça marchera direct.
            # Sinon, on peut copier le fichier vers le dossier attendu par la lib :
            home_hsemotion = os.path.expanduser("~/.hsemotion")
            os.makedirs(home_hsemotion, exist_ok=True)
            dest_path = os.path.join(home_hsemotion, "enet_b0_8_va_mtl.pt")
            
            if not os.path.exists(dest_path):
                import shutil
                print("Copie du modèle vers le répertoire utilisateur pour hsemotion...")
                shutil.copy(model_path, dest_path)
            
            self.model = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl', device=device)
        else:
            print(f"ERREUR : Le fichier {model_path} est introuvable.")
            # Tentative de secours standard
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