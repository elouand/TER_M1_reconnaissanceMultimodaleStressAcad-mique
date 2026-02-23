import socket
import wave
import numpy as np
import noisereduce as nr

UDP_IP = "0.0.0.0"
UDP_PORT = 5006

# Augmenter la taille du buffer pour accepter les gros paquets de Pepper
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Réception en cours... (Parlez au robot, puis faites Ctrl+C pour sauvegarder)")

frames = []

try:
    while True:
        # 65535 est la taille max possible pour un paquet UDP
        data, addr = sock.recvfrom(65535)
        frames.append(data)
        # Petit indicateur visuel pour savoir que ça travaille
        print(".", end="", flush=True) 
except KeyboardInterrupt:
    print("\nEnregistrement terminé.")
finally:
    if frames:
        # On convertit les données binaires en tableau de nombres (Int16)
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
        
        # Pepper envoie 4 canaux. On ne garde que le premier (index 0)
        # On prend 1 échantillon sur 4
        print(audio_data.shape)
        mono_audio = audio_data[0::4] 
        print(mono_audio.shape)
        
        filename = "voix_humaine.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1) # On passe en Mono
            wf.setsampwidth(2)
            wf.setframerate(48000) # Teste 16000 ou 48000 si c'est encore trop grave
            wf.writeframes(mono_audio.tobytes())
        print(f"Fichier corrigé sauvegardé : {filename}")