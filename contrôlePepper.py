# -*- coding: utf-8 -*-
import sys
import tty
import termios
from naoqi import ALProxy

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main(robot_ip="127.0.0.1"):
    try:
        motion = ALProxy("ALMotion", robot_ip, 9559)
        motion.wakeUp()
        
        # Bypass de la sécurité (pour corriger les LEDs jaunes)
        motion.setExternalCollisionProtectionEnabled("All", False)
        motion.setStiffnesses("Arms", 1.0)
        
        # Position de départ du bras (verticale basse)
        current_pitch = 1.5 

        print("\n--- PILOTAGE TEMPS RÉEL ACTIVÉ ---")
        print("Z / S : Avancer / Reculer")
        print("Q / D : Gauche / Droite")
        print("J / L : Pivoter")
        print("I / K : LEVER / BAISSER LE BRAS")
        print("Espace : STOP | CTRL+C : Quitter")
        print("----------------------------------\n")

        while True:
            key = get_key().lower()
            
            # --- DÉPLACEMENTS (ROUES) ---
            if key == 'z':
                motion.post.moveTo(0.2, 0.0, 0.0)
            elif key == 's':
                motion.post.moveTo(-0.2, 0.0, 0.0)
            elif key == 'q':
                motion.post.moveTo(0.0, 0.1, 0.0)
            elif key == 'd':
                motion.post.moveTo(0.0, -0.1, 0.0)
            elif key == 'j':
                motion.post.moveTo(0.0, 0.0, 0.5)
            elif key == 'l':
                motion.post.moveTo(0.0, 0.0, -0.5)
            
            # --- CONTRÔLE DU BRAS (I / K) ---
            elif key == 'i':
                current_pitch -= 0.2 # On monte
                if current_pitch < -1.5: current_pitch = -1.5 # Limite haute
                motion.setAngles("RShoulderPitch", current_pitch, 0.2)
                
            elif key == 'k':
                current_pitch += 0.2 # On baisse
                if current_pitch > 1.5: current_pitch = 1.5 # Limite basse
                motion.setAngles("RShoulderPitch", current_pitch, 0.2)

            # --- ARRÊT ET SORTIE ---
            elif key == ' ':
                motion.stopMove()
            elif key == '\x03': # CTRL+C pour quitter
                break
                
    except Exception as e:
        print "Erreur :", e
    finally:
        motion.stopMove()
        # On réactive la sécurité avant de partir, c'est plus prudent !
        motion.setExternalCollisionProtectionEnabled("All", True)
        print("\nArrêt du programme. Sécurités réactivées.")

if __name__ == "__main__":
    main()