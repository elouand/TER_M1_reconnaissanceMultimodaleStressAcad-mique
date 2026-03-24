# -*- coding: utf-8 -*-
import sys
import tty
import termios
import time
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
    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)

    try:
        motion = ALProxy("ALMotion", robot_ip, 9559)
        tts = ALProxy("ALTextToSpeech", robot_ip, 9559)
        life = ALProxy("ALAutonomousLife", robot_ip, 9559)
        
        # Désactivation des mouvements autonomes pour garder la tablette active
        try:
            life.setAutonomousAbilityEnabled("BasicAwareness", False)
            life.setAutonomousAbilityEnabled("BackgroundMovement", False)
            motion.setBreathEnabled("All", False)
        except Exception as e:
            print("[!] Note: Erreur init :", e)

        motion.wakeUp()
        motion.setStiffnesses(["RArm", "Head", "Move"], 1.0)
        
        arm_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]
        head_names = ["HeadYaw", "HeadPitch"]
        
        # Votre scénario (inchangé pour garder vos données)
        scenario = [
            [[0.6627, -0.0445, 2.0847, 0.0997, -0.0553, 0.9112], [-0.1396, -0.1074], "Etape 1"],
            [[0.6627, -0.0445, 2.0847, 0.0997, -0.0553, 0.0], [-0.1396, -0.1074], "TOUCHE BAS"],
            [[0.6642, -0.4893, 2.0801, 0.0951, -0.0721, 0.0404], [-0.0966, -0.1028], "goSide"],
            [[0.158, -0.8007, 2.0847, 0.0982, -0.0599, 0.0413], [-0.1396, -0.1304], "Etape 3"],
            [[0.158, -0.8007, 2.0847, 0.0982, -0.0599, 1.0], [-0.1396, -0.1304], "Etape 4"],
            [[0.0506, -0.201, 2.0847, 0.0107, -0.0844, 0.9112], [-0.1381, -0.1304], "un peu en haut"],
            [[0.3083, -0.1503, 0.0782, 0.0123, -0.0599, 0.9042], [-0.1381, -0.1304], "TOUCHE HAUT"],
            [[0.0506, -0.201, 2.0847, 0.0107, -0.0844, 0.9112], [-0.1381, -0.1304], "un peu en haut"],
            [[0.158, -0.8007, 2.0847, 0.0982, -0.0599, 1.0], [-0.1396, -0.1304], "retour vers départ"],
            [[0.6642, -0.4893, 2.0801, 0.0951, -0.0721, 0.0404], [-0.0966, -0.1028], "goSide"],
        ]
        
        step_idx = 0
        # On initialise les angles actuels
        actual_arm = motion.getAngles("RArm", True)
        actual_head = motion.getAngles("Head", True)
        angles = dict(zip(arm_names, actual_arm))
        angles.update(dict(zip(head_names, actual_head)))
        
        move_step = 0.05 

        print("\n============================================")
        print("      CONTROLE PEPPER (BRAS ISOLE)          ")
        print("============================================")
        print(" N : Bouger BRAS uniquement | R : Reset")
        print(" IJKL : Bouger TETE manuellement")
        print(" A / E : Rotation Gauche / Droite")
        print(" Z S Q D : Deplacements base")
        print("============================================\n")

        while True:
            key = get_key().lower()
            
            # --- LOGIQUE SCENARIO (MODIFIÉE) ---
            if key == 'n':
                if step_idx < len(scenario):
                    a_vals, h_vals, msg = scenario[step_idx]
                    # On n'envoie que les angles du bras (arm_names)
                    motion.post.setAngles(arm_names, a_vals, 0.1)
                    print("[>] Etape %d : %s (Bras uniquement)" % (step_idx + 1, msg))
                    step_idx += 1

            elif key == 'r':
                step_idx = 0
                print("\n[!] Scenario remis a zero.")

            # --- MANUEL : BRAS (PAVE NUM) ---
            elif key == '8': angles["RShoulderPitch"] -= move_step; motion.post.setAngles("RShoulderPitch", angles["RShoulderPitch"], 0.1)
            elif key == '2': angles["RShoulderPitch"] += move_step; motion.post.setAngles("RShoulderPitch", angles["RShoulderPitch"], 0.1)
            elif key == '4': angles["RShoulderRoll"] -= move_step; motion.post.setAngles("RShoulderRoll", angles["RShoulderRoll"], 0.1)
            elif key == '6': angles["RShoulderRoll"] += move_step; motion.post.setAngles("RShoulderRoll", angles["RShoulderRoll"], 0.1)
            elif key == '7': angles["RElbowRoll"] += move_step; motion.post.setAngles("RElbowRoll", angles["RElbowRoll"], 0.1)
            elif key == '9': angles["RElbowRoll"] -= move_step; motion.post.setAngles("RElbowRoll", angles["RElbowRoll"], 0.1)
            elif key == '1': angles["RElbowYaw"] -= move_step; motion.post.setAngles("RElbowYaw", angles["RElbowYaw"], 0.1)
            elif key == '3': angles["RElbowYaw"] += move_step; motion.post.setAngles("RElbowYaw", angles["RElbowYaw"], 0.1)
            elif key == '5': motion.post.setAngles("RHand", 1.0, 0.2)
            elif key == '0': motion.post.setAngles("RHand", 0.0, 0.2)

            # --- MANUEL : TETE (IJKL) ---
            elif key == 'i': angles["HeadPitch"] -= move_step; motion.post.setAngles("HeadPitch", angles["HeadPitch"], 0.1)
            elif key == 'k': angles["HeadPitch"] += move_step; motion.post.setAngles("HeadPitch", angles["HeadPitch"], 0.1)
            elif key == 'j': angles["HeadYaw"] += move_step; motion.post.setAngles("HeadYaw", angles["HeadYaw"], 0.1)
            elif key == 'l': angles["HeadYaw"] -= move_step; motion.post.setAngles("HeadYaw", angles["HeadYaw"], 0.1)

            # --- ROUES ---
            elif key == 'z': motion.post.moveTo(0.2, 0.0, 0.0)
            elif key == 's': motion.post.moveTo(-0.2, 0.0, 0.0)
            elif key == 'q': motion.post.moveTo(0.0, 0.1, 0.0)
            elif key == 'd': motion.post.moveTo(0.0, -0.1, 0.0)
            elif key == 'a': motion.post.moveTo(0.0, 0.0, 0.15) 
            elif key == 'e': motion.post.moveTo(0.0, 0.0, -0.15) 

            elif key == 'y':
                termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
                v_arm = [round(a, 4) for a in motion.getAngles("RArm", True)]
                v_head = [round(a, 4) for a in motion.getAngles("Head", True)]
                print("\n[%s, %s, \"Message\"]," % (v_arm, v_head))
                tty.setraw(fd)

            elif key == 't':
                termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
                phrase = raw_input("\nDire : "); tts.post.say(phrase); tty.setraw(fd)
            elif key == ' ': motion.stopMove()
            elif key == '\x03': break

            # Sécurités
            angles["HeadPitch"] = max(-0.6, min(0.6, angles["HeadPitch"]))
            angles["RShoulderRoll"] = max(-1.5, min(-0.1, angles["RShoulderRoll"]))

    except Exception as e: 
        print "\nErreur :", e
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
        motion.stopMove()
        try:
            life.setAutonomousAbilityEnabled("BasicAwareness", True)
            life.setAutonomousAbilityEnabled("BackgroundMovement", True)
            motion.setBreathEnabled("All", True)
        except: pass
        print("\nFermeture.")

if __name__ == "__main__":
    main()