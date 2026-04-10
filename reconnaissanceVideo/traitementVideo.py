import torch
import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from hsemotion.facial_emotions import HSEmotionRecognizer

# =========================================================
# 🛡️ PATCH DE SÉCURITÉ PYTORCH 2.6+ (OBLIGATOIRE)
# =========================================================
import torch.serialization
original_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # On force weights_only à False pour autoriser le modèle HSEmotion
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = _patched_torch_load
# =========================================================

class FaceAligner:
    def __init__(self, model_path='face_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def align_and_crop(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None, None

        landmarks = detection_result.face_landmarks[0]
        h, w, _ = frame_rgb.shape
        
        # Yeux : 33 (gauche), 263 (droit)
        left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
        right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])

        dY, dX = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_img = cv2.warpAffine(frame_rgb, M, (w, h), flags=cv2.INTER_CUBIC)

        dist_eyes = np.linalg.norm(right_eye - left_eye)
        size = int(dist_eyes * 1.8) # Zone de crop
        x1, y1 = max(0, center[0]-size), max(0, center[1]-size)
        face_crop = aligned_img[y1:center[1]+size, x1:center[0]+size]
        
        return face_crop, center

class EmotionRegressor:
    def __init__(self, device='cpu'):
        # Chemin vers face_landmarker.task
        dir_path = os.path.dirname(os.path.abspath(__file__))
        task_path = os.path.join(dir_path, 'face_landmarker.task')
        self.aligner = FaceAligner(model_path=task_path)
        
        # On utilise le B0 car le B2 est introuvable sur les serveurs
        self.model_name = 'enet_b0_8_va_mtl' 
        print(f"🔄 Chargement de {self.model_name}...")
        
        # Grâce au patch ci-dessus, cette ligne ne plantera plus !
        self.model = HSEmotionRecognizer(model_name=self.model_name, device=device)
        
        self.last_va = np.array([0.0, 0.0])
        self.alpha = 0.3 

    def get_vad(self, frame_rgb):
        face_crop, center = self.aligner.align_and_crop(frame_rgb)
        
        if face_crop is None or face_crop.size == 0:
            return np.array([self.last_va[0], self.last_va[1], (self.last_va[1] + abs(self.last_va[0])) / 2])

        # Resize 224 pour le modèle B0
        face_resized = cv2.resize(face_crop, (224, 224))
        face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
        
        _, va = self.model.predict_emotions(face_bgr, logits=True)
        
        v_raw, a_raw = -va[0], -va[1]
        new_va = np.array([np.tanh(v_raw), np.tanh(a_raw)])
        
        # Lissage
        self.last_va = self.alpha * new_va + (1 - self.alpha) * self.last_va
        
        dom = (self.last_va[1] + abs(self.last_va[0])) / 2
        return np.array([self.last_va[0], self.last_va[1], dom])