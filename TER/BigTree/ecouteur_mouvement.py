# -*- coding: utf-8 -*-

import socket
from naoqi import ALProxy

UDP_IP = "0.0.0.0"
PORT_MOTION = 5009

def main():
    motion = ALProxy("ALMotion", "127.0.0.1", 9559)
    # Assouplir pour éviter que le robot ne soit trop raide, mais garder la position
    motion.setStiffnesses("RArm", 0.8)
    
    # Noms des articulations
    arm_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]
    
    # On définit un pas (plus grand que 0.05 pour que ce soit bien visible)
    move_step = 0.25 

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, PORT_MOTION))
    print("🤖 Pepper prêt : Écoute des commandes de bras sur le port {}".format(PORT_MOTION))

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            cmd = data.strip()
            
            # Récupérer la position actuelle
            cur_arm = motion.getAngles("RArm", True)
            arm_dict = dict(zip(arm_names, cur_arm))
            
            # Flexion (7) et Extension (8) du coude
            if cmd == "7":
                arm_dict["RElbowRoll"] += move_step
                print(" -> Flexion")
            elif cmd == "8":
                arm_dict["RElbowRoll"] -= move_step
                print(" -> Extension")
                
            # Exécution rapide (0.15 seconde)
            motion.setAngles(arm_names, [arm_dict[name] for name in arm_names], 0.15)
            
    except KeyboardInterrupt:
        print("\nArrêt de l'écouteur.")

if __name__ == "__main__":
    main()
