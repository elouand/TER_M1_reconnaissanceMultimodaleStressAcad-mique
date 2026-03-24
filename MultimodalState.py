import numpy as np 
import time

class MultimodalState:
    def __init__(self):
        # Initialisation : [V, A, D, Timestamp]
        # On utilise 0.0 partout au début pour un état neutre
        self.data = {
            "vision": np.array([0.0, 0.0, 0.0, time.time()]),
            "audio":  np.array([0.0, 0.0, 0.0, time.time()]),
            "texte":  np.array([0.0, 0.0, 0.0, 0.0]) # Timestamp 0 = périmé d'office
        }

        # MATRICE DE POIDS (Vecteur [W_v, W_a, W_d] par modalité)
        self.weights = {
            "vision": np.array([0.8, 0.4, 0.3]), # Expert en Valence
            "audio":  np.array([0.4, 0.9, 0.5]), # Expert en Arousal (Stress)
            "texte":  np.array([0.9, 0.2, 0.9])  # Expert en Valence/Dominance
        }

        # Constantes d'amortissement (lambda)
        self.decay = {"vision": 0.01, "audio": 0.15, "texte": 0.1}

    def update(self, modality, vad_values):
        """
        vad_values: list ou np.array de 3 éléments [V, A, D]
        """
        try:
            # On force la conversion en float au cas où
            v, a, d = vad_values
            # On reconstruit un array propre de taille 4
            self.data[modality] = np.array([float(v), float(a), float(d), time.time()])
        except (ValueError, TypeError) as e:
            print(f"[ERREUR FUSION] Format VAD incorrect pour {modality}: {vad_values}")

    def get_fusion(self):
        now = time.time()
        numerator = np.zeros(3)
        denominator = np.zeros(3)

        for m in self.data:
            # Extraction sécurisée des 3 premières valeurs (VAD) et de la 4ème (Time)
            current_entry = self.data[m]
            vad = current_entry[:3]
            t_last = current_entry[3]
            
            # Calcul de l'amortissement temporel
            dt = now - t_last
            # On limite dt pour éviter des overflow avec exp
            alpha = np.exp(-self.decay[m] * min(dt, 3600)) 
            
            # Application des poids matriciels pondérés par l'amortissement
            current_w = self.weights[m] * alpha
            
            numerator += vad * current_w
            denominator += current_w

        # Division par dimension pour normaliser
        # On évite la division par zéro avec un petit epsilon
        final_vad = numerator / (denominator + 1e-6)
        
        # HSEmotion et ton unifcation utilisent -1 à 1
        return np.clip(final_vad, -1.0, 1.0)