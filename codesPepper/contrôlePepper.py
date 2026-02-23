# -*- coding: utf-8 -*-
import sys
import tty
import termios
import time
from naoqi import ALProxy

def get_key():
    """ Capture une touche du clavier sans attendre Entrée """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main(robot_ip="127.0.0.1"):
    # Sauvegarde de la configuration du terminal pour le mode texte
    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)

    try:
        # Initialisation des services
        motion = ALProxy("ALMotion", robot_ip, 9559)
        tts = ALProxy("ALTextToSpeech", robot_ip, 9559)
        
        print("Preparation du robot...")
        tts.setLanguage("French")
        motion.wakeUp()
        
        # Désactivation des sécurités de collision (évite le blocage LED Jaune)
        motion.setExternalCollisionProtectionEnabled("All", False)
        motion.setStiffnesses("Body", 1.0)
        
        current_pitch = 1.0 # Position de base du bras

        print("\n============================================")
        print("      CONTROLE PEPPER - VERSION ULTIME      ")
        print("============================================")
        print(" Z / S : Avancer / Reculer")
        print(" Q / D : Gauche / Droite")
        print(" J / L : Rotation Gauche / Droite")
        print("--------------------------------------------")
        print(" I / K : Lever / Baisser le bras (VITESSE MAX)")
        print(" T     : PARLER (Ouvre la saisie de texte)")
        print(" 8     : COUP DE POING (PUNCH)")
        print("--------------------------------------------")
        print(" 1 : Pierre | 2 : Feuille | 3 : Ciseaux (V-Shape)")
        print("--------------------------------------------")
        print(" ESPACE : STOP | CTRL+C : QUITTER")
        print("============================================\n")

        while True:
            key = get_key().lower()
            
            # --- DEPLACEMENTS (ROUES) ---
            if key == 'z': motion.post.moveTo(0.2, 0.0, 0.0)
            elif key == 's': motion.post.moveTo(-0.2, 0.0, 0.0)
            elif key == 'q': motion.post.moveTo(0.0, 0.1, 0.0)
            elif key == 'd': motion.post.moveTo(0.0, -0.1, 0.0)
            elif key == 'j': motion.post.moveTo(0.0, 0.0, 0.5)
            elif key == 'l': motion.post.moveTo(0.0, 0.0, -0.5)
            
            # --- BRAS MANUEL (VITESSE 0.8) ---
            elif key == 'i':
                current_pitch -= 0.4
                if current_pitch < -1.5: current_pitch = -1.5
                motion.setAngles("RShoulderPitch", current_pitch, 0.8)
            elif key == 'k':
                current_pitch += 0.4
                if current_pitch > 1.5: current_pitch = 1.5
                motion.setAngles("RShoulderPitch", current_pitch, 0.8)

            # --- LE PUNCH (TOUCHE 8) ---
            elif key == '8':
                print("[Action] Coup de poing !")
                tts.post.say("Prends ça !")
                # Armer le bras
                motion.setAngles(["RHand", "RShoulderPitch", "RElbowRoll", "RShoulderRoll"], [0.0, 0.5, 1.5, -0.2], 0.6)
                time.sleep(0.4)
                # Frapper (Vitesse 1.0)
                motion.setAngles(["RShoulderPitch", "RElbowRoll"], [-0.2, 0.1], 1.0)
                time.sleep(0.3)
                # Retour
                motion.setAngles(["RShoulderPitch", "RElbowRoll", "RShoulderRoll"], [1.0, 0.5, -0.2], 0.5)

            # --- PARLER (TOUCHE T) ---
            elif key == 't':
                termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
                print("\n[Vocal] Entrez votre phrase : "),
                phrase = raw_input()
                tts.post.say(phrase)
                tty.setraw(fd)

            # --- JEU : PIERRE / FEUILLE / CISEAUX ---
            elif key == '1': # PIERRE
                motion.setAngles(["RHand", "RWristYaw", "RShoulderPitch"], [0.0, 0.0, 0.5], 0.8)
                tts.post.say("Pierre !")
            
            elif key == '2': # FEUILLE
                motion.setAngles(["RHand", "RWristYaw", "RShoulderPitch", "RElbowRoll"], [1.0, 0.0, 0.0, 0.0], 0.8)
                tts.post.say("Feuille !")
            
            elif key == '3': # CISEAUX (Astuce visuelle : main sur la tranche)
                # Pepper ne peut pas isoler l'index et le majeur (1 seul moteur pour la main)
                # On tourne le poignet a 90 degres pour simuler la forme
                motion.setAngles(["RHand", "RWristYaw", "RShoulderPitch", "RElbowRoll"], [1.0, 1.5, 0.2, 0.6], 0.8)
                tts.post.say("Ciseaux !")
                time.sleep(0.1)
                motion.setAngles("RElbowRoll", 0.3, 0.8) # Petit mouvement de coupe
                time.sleep(0.1)
                motion.setAngles("RElbowRoll", 0.6, 0.8)

            # --- STOP ET QUITTER ---
            elif key == ' ':
                motion.stopMove()
            elif key == '\x03': # Ctrl+C
                break
                
    except Exception as e:
        print "\nErreur rencontree :", e
    finally:
        # Restauration systematique du terminal et des securites
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
        motion.stopMove()
        motion.setExternalCollisionProtectionEnabled("All", True)
        print("\nProgramme arrete. Securites Pepper reactivees.")

if __name__ == "__main__":
    main()