 
# CODE CLIENT SUR LA MACHINE A LANCER AVANT DE LANCER LE CODE ROBOT

import socket
import cv2
import numpy as np

# Configuration
UDP_IP = "0.0.0.0" # Écoute sur toutes les interfaces
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Serveur d'écoute d'émotions lancé...")

while True:
    # Réception des données
    data, addr = sock.recvfrom(65507) 
    
    # Décodage de l'image
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        # --- ICI VOUS APPELEZ VOTRE MODÈLE D'IA ---
        # exemple: emotion = model.predict(frame)
        
        cv2.imshow("Flux Pepper - Detection Emotion", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sock.close()