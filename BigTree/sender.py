#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import cv2
import numpy as np
from naoqi import ALProxy
import time
import signal
import sys

# Remplacez par l'IP de votre PC
PC_IP = "169.254.172.13" 
PORT = 5005

def main():
    try:
        video_proxy = ALProxy("ALVideoDevice", "127.0.0.1", 9559)
    except Exception as e:
        print("Erreur proxy: " + str(e))
        return

    def manual_stop(sig, frame):
        video_proxy.unsubscribe(name_id)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, manual_stop)

    # 0: TopCam, 1: QVGA (320x240), 11: RGB, 5: FPS
    name_id = video_proxy.subscribeCamera("client", 0, 1, 11, 5)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("Envoi vers " + PC_IP + "...")

    video_proxy.setFrameRate(name_id, 20)	

    try:
        while True:
            result = video_proxy.getImageRemote(name_id)
            if result is None:
                continue

            width = result[0]
            height = result[1]
            # Transformation des donnees brutes
            img = np.frombuffer(result[6], dtype=np.uint8).reshape((height, width, 3))

            # Compression JPEG a 50% pour alleger le paquet UDP
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buf = cv2.imencode('.jpg', img, params)

            # Envoi si la taille est inferieure a la limite UDP (65507 octets)
            if len(buf) < 65507:
                sock.sendto(buf, (PC_IP, PORT))
            else:
                print("Image trop grosse")

    except KeyboardInterrupt:
        print("Arret...")
    finally:
        video_proxy.unsubscribe(name_id)
        sock.close()

if __name__ == "__main__":
    main()
