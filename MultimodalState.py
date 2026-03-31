# -*- coding: utf-8 -*-
import numpy as np 
import time

class MultimodalState:
    def __init__(self):
        # Etat interne : [Valence, Arousal]
        self.current_fusion = np.array([0.0, 0.0])
        
        # Buffers pour l'alignement (timestamp, np.array([V, A]))
        self.buffer_vision = []
        self.buffer_audio = []
        
        # Dernières valeurs reçues (pour affichage temps réel)
        self.data = {
            "vision": np.array([0.0, 0.0]),
            "audio":  np.array([0.0, 0.0]),
            "texte":  np.array([0.0, 0.0])
        }

        # MATRICE DE POIDS VA (Vecteur [W_v, W_a] par modalité)
        self.weights = {
            "vision": np.array([0.8, 0.4]), # Expert Valence (Sourire)
            "audio":  np.array([0.4, 0.9]), # Expert Arousal (Stress/Volume)
            "texte":  np.array([0.9, 0.3])  # Expert Valence (Mots-clés)
        }

        self.MAX_BUFFER_TIME = 10 # 10 secondes de mémoire vive

    def update(self, modality, va_values):
        now = time.time()
        # On ne garde que les 2 premières valeurs (V, A)
        v_array = np.array([float(va_values[0]), float(va_values[1])])
        
        self.data[modality] = v_array

        if modality == "vision":
            self.buffer_vision.append((now, v_array))
        elif modality == "audio":
            self.buffer_audio.append((now, v_array))

        # Nettoyage des buffers
        self.buffer_vision = [x for x in self.buffer_vision if now - x[0] < self.MAX_BUFFER_TIME]
        self.buffer_audio = [x for x in self.buffer_audio if now - x[0] < self.MAX_BUFFER_TIME]

    def get_synced_fusion(self, target_time):
        """ Aligne la Vision et l'Audio sur le moment X du Texte """
        def find_closest(buffer, t):
            if not buffer: return np.array([0.0, 0.0])
            closest = min(buffer, key=lambda x: abs(x[0] - t))
            return closest[1]

        v_v = find_closest(self.buffer_vision, target_time)
        v_a = find_closest(self.buffer_audio, target_time)
        v_t = self.data["texte"]

        # Calcul pondéré VA
        num = (v_v * self.weights["vision"]) + \
              (v_a * self.weights["audio"]) + \
              (v_t * self.weights["texte"])
              
        den = self.weights["vision"] + self.weights["audio"] + self.weights["texte"]

        self.current_fusion = np.clip(num / (den + 1e-6), -1.0, 1.0)
        return self.current_fusion

    def get_fusion(self):
        """ Retourne la vision en temps réel pour l'UI """
        return self.data["vision"]