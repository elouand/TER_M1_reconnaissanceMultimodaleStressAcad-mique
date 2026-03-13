import numpy as np 
import time

class MultimodalState:
    def __init__(self):
        # Initialisation : [V, A, D, Timestamp]
        self.data = {
            "vision": np.array([0.5, 0.5, 0.5, time.time()]),
            "audio":  np.array([0.5, 0.5, 0.5, time.time()]),
            "texte":  np.array([0.5, 0.5, 0.5, 0.0])
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
        # vad_values doit être un array/liste de 3 éléments
        self.data[modality] = np.append(vad_values, time.time())

    def get_fusion(self):
        now = time.time()
        numerator = np.zeros(3)
        denominator = np.zeros(3)

        for m in self.data:
            # Récupération des valeurs et du temps
            vad = self.data[m][:3]
            t_last = self.data[m][3]
            
            # Calcul de l'amortissement temporel
            dt = now - t_last
            alpha = np.exp(-self.decay[m] * dt)
            
            # Application des poids matriciels pondérés par l'amortissement
            current_w = self.weights[m] * alpha
            
            numerator += vad * current_w
            denominator += current_w

        # Division par dimension pour normaliser
        # On évite la division par zéro avec un petit epsilon
        final_vad = numerator / (denominator + 1e-6)
        return np.clip(final_vad, -1, 1) # Ou 0, 1 selon ton modèle