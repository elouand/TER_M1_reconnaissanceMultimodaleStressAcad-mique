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
                "audio":  np.array([0.4, 0.9]), # Expert Arousal (Volume/Ton)
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
            """ Aligne la Vision et l'Audio avec interpolation """
            def get_interpolated_value(buffer, t):
                if not buffer: return np.array([0.0, 0.0])
                if len(buffer) == 1: return buffer[0][1]
                
                buffer.sort(key=lambda x: x[0])
                
                for i in range(len(buffer) - 1):
                    if buffer[i][0] <= t <= buffer[i+1][0]:
                        t1, v1 = buffer[i]
                        t2, v2 = buffer[i+1]
                        return v1 + (v2 - v1) * ((t - t1) / (t2 - t1))
                
                return min(buffer, key=lambda x: abs(x[0] - t))[1]

            v_v = get_interpolated_value(self.buffer_vision, target_time)
            v_a = get_interpolated_value(self.buffer_audio, target_time)
            v_t = self.data["texte"] 

            # Calcul VA fusionné pondéré
            num = (v_v * self.weights["vision"]) + (v_a * self.weights["audio"]) + (v_t * self.weights["texte"])
            den = self.weights["vision"] + self.weights["audio"] + self.weights["texte"]

            self.current_fusion = np.clip(num / (den + 1e-6), -1.0, 1.0)
            return self.current_fusion

        def get_fusion(self):
            """ Retourne la fusion TEMPS RÉEL (Vision + Audio) pour l'UI """
            v_v = self.data["vision"]
            v_a = self.data["audio"]
            
            # On fusionne Vision et Audio pour le graphique (plus réactif que Vision seule)
            num = (v_v * self.weights["vision"]) + (v_a * self.weights["audio"])
            den = self.weights["vision"] + self.weights["audio"]
            
            return np.clip(num / (den + 1e-6), -1.0, 1.0)


""" 
        def get_synced_fusion(self, target_time):
            # Aligne la Vision et l'Audio sur le moment X du Texte 
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
            # Retourne la vision en temps réel pour l'UI 
            return self.data["vision"]
        """