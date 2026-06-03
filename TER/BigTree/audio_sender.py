# -*- coding: utf-8 -*-
import socket
import numpy as np
from naoqi import ALProxy, ALModule, ALBroker
import time

# Configuration réseau
PC_IP = "169.254.172.13"
PORT_AUDIO = 5006
PEPPER_IP = "127.0.0.1"

class AudioRawModule(ALModule):
    def __init__(self, name):
        ALModule.__init__(self, name)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timestamp, buff):
        """ Callback appelé par Naoqi toutes les 16ms """
        try:
            # CORRECTION : on utilise bien la variable 'buff'
            self.sock.sendto(buff, (PC_IP, PORT_AUDIO))
        except Exception as e:
            pass # On évite de spammer la console si un paquet saute

def main():
    myBroker = ALBroker("myBroker", "0.0.0.0", 0, PEPPER_IP, 9559)
    
    global AudioModule
    AudioModule = AudioRawModule("AudioModule")
    audio_proxy = ALProxy("ALAudioDevice")

    # --- CONFIGURATION DU MICRO SELECTIONNE ---
    # 48000 Hz, Canal 3 (Micro Avant), 0 (Pas de post-traitement)
    print("Configuration du micro AVANT (Canal 3)...")
    audio_proxy.setClientPreferences("AudioModule", 16000, 3, 0)

    print("Tentative de souscription...")
    audio_proxy.subscribe("AudioModule")
    print("Flux Audio Mono actif vers " + PC_IP + " (Port 5006)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArrêt propre...")
        audio_proxy.unsubscribe("AudioModule")
        myBroker.shutdown()

if __name__ == "__main__":
    main()
