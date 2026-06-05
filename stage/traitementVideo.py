import torch
import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn.functional as F

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
        
        left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
        right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])

        dY, dX = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_img = cv2.warpAffine(frame_rgb, M, (w, h), flags=cv2.INTER_CUBIC)

        dist_eyes = np.linalg.norm(right_eye - left_eye)
        size = int(dist_eyes * 1.8)
        x1, y1 = max(0, center[0]-size), max(0, center[1]-size)
        face_crop = aligned_img[y1:center[1]+size, x1:center[0]+size]
        
        return face_crop, center

class EmotionRegressor:
    def __init__(self, device='cpu'):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        task_path = os.path.join(dir_path, 'face_landmarker.task')
        self.aligner = FaceAligner(model_path=task_path)
        self.device = device
        
        print("Chargement du Vision Transformer (ViT) via Hugging Face...")
        self.model_name = 'dima806/facial_emotions_image_detection'
        
        # Initialisation propre via l'écosystème Transformers
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTForImageClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        self.last_va = np.array([0.0, 0.0])
        self.alpha = 0.3 

        # Cartographie Psychologique (Modèle de Russell)
        # Chaque émotion détectée "tire" la Valence et l'Arousal vers ces coordonnées
        self.emotion_coords = {
            'angry':    np.array([-0.8,  0.8]),
            'disgust':  np.array([-0.8,  0.4]),
            'fear':     np.array([-0.6,  0.8]),
            'happy':    np.array([ 0.9,  0.5]),
            'neutral':  np.array([ 0.0,  0.0]),
            'sad':      np.array([-0.9, -0.5]),
            'surprise': np.array([ 0.2,  0.9])
        }

    def get_vad(self, frame_rgb):
        face_crop, center = self.aligner.align_and_crop(frame_rgb)
        
        if face_crop is None or face_crop.size == 0:
            return np.array([self.last_va[0], self.last_va[1], (self.last_va[1] + abs(self.last_va[0])) / 2])

        # Préparation de l'image pour le Transformer (Resize 224x224, Normalisation RGB)
        inputs = self.processor(images=face_crop, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Plus tard, pour ton LSTM, tu pourras récupérer outputs.hidden_states ici !

        # Calcul des probabilités de chaque émotion (Softmax)
        probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        
        # Calcul de l'Espérance Mathématique 2D
        v_calc, a_calc = 0.0, 0.0
        for i, prob in enumerate(probs):
            label = self.model.config.id2label[i].lower()
            coords = self.emotion_coords.get(label, np.array([0.0, 0.0]))
            v_calc += prob * coords[0]
            a_calc += prob * coords[1]

        # Lissage temporel basique (EMA) inter-frames
        new_va = np.array([v_calc, a_calc])
        self.last_va = self.alpha * new_va + (1 - self.alpha) * self.last_va
        
        dom = (self.last_va[1] + abs(self.last_va[0])) / 2
        return np.array([round(self.last_va[0], 4), round(self.last_va[1], 4), round(dom, 4)])