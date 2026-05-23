import cv2

def preprocess_video(video_path, target_fps=5):#Récupère la vidéo envoyée par Pepper et la pré-traite pour envoyer à traitementVideo
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    hop_length = max(1, int(fps / target_fps))
    
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % hop_length == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frames.append(frame_resized)
        count += 1
    cap.release()
    return frames