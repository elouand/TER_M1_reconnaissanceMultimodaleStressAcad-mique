# Au lieu de self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(...)
self.face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)