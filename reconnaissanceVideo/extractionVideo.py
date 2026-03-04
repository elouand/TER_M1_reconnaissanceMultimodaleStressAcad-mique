import cv2

def preprocess_video(video_path, target_fps=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    hop_length = int(fps / target_fps)
    
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % hop_length == 0:
            # 1. Détection de visage (ex: via MediaPipe ou MTCNN)
            # 2. Crop & Resize (ex: 224x224 pour EfficientNet)
            # 3. Normalisation
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        count += 1
    cap.release()
    return frames