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

        # Augmentation de la mémoire vive à 30 secondes pour permettre 
        # le découpage rétroactif de longues phrases sans perdre les données du début.
        self.MAX_BUFFER_TIME = 45

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
        """ Aligne la Vision et l'Audio sur le moment X du Texte (Ancienne méthode ponctuelle) """
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

    def get_detailed_state_window(self, start_w, end_w):
        """ Retourne la MOYENNE [Valence, Arousal] sur une fenêtre de temps, et recalcule la fusion """
        
        def get_window_average(buffer, start_t, end_t):
            # 1. On filtre les éléments capturés pendant la tranche de temps (ex: 2 secondes)
            in_window = [item[1] for item in buffer if start_t <= item[0] <= end_t]
            
            if in_window:
                # 2. Moyenne mathématique sur la colonne Valence et la colonne Arousal
                return np.mean(in_window, axis=0)
            else:
                # 3. Sécurité : Si aucun log dans cette tranche (ex: perte de frame caméra), 
                # on prend la dernière valeur connue juste avant la tranche.
                before = [item for item in buffer if item[0] <= end_t]
                if before:
                    return before[-1][1]
                return np.array([0.0, 0.0])

        # Récupération des moyennes [Valence, Arousal]
        v_v = get_window_average(self.buffer_vision, start_w, end_w)
        v_a = get_window_average(self.buffer_audio, start_w, end_w)
        v_t = self.data["texte"] # Le texte est constant pour la phrase, on prend la valeur actuelle

        # Recalcul de la fusion pondérée avec ces moyennes
        num = (v_v * self.weights["vision"]) + \
              (v_a * self.weights["audio"]) + \
              (v_t * self.weights["texte"])
              
        den = self.weights["vision"] + self.weights["audio"] + self.weights["texte"]

        fusion = np.clip(num / (den + 1e-6), -1.0, 1.0)
        
        return fusion, v_v, v_a, v_t

    def get_fusion(self):
        """ Retourne la vision en temps réel pour l'UI """
        return self.data["vision"]