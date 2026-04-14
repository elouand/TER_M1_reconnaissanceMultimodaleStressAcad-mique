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
        
        scenario = [
            [[0.6627, -0.0445, 2.0847, 0.0997, -0.0553, 0.9112], [-0.1396, -0.1074], "Etape 1"],
            [[0.6627, -0.0445, 2.0847, 0.0997, -0.0553, 0.0], [-0.1396, -0.1074], "TOUCHE BAS"],
            [[0.6642, -0.4893, 2.0801, 0.0951, -0.0721, 0.0404], [-0.0966, -0.1028], "goSide"],
            [[0.158, -0.8007, 2.0847, 0.0982, -0.0599, 0.0413], [-0.1396, -0.1304], "Etape 3"],
            [[0.158, -0.8007, 2.0847, 0.0982, -0.0599, 1.0], [-0.1396, -0.1304], "Etape 4"],
            [[0.0506, -0.201, 2.0847, 0.0107, -0.0844, 0.9112], [-0.1381, -0.1304], "un peu en haut"],
            [[0.3083, -0.1503, 0.0782, 0.0123, -0.0599, 0.9042], [-0.1381, -0.1304], "TOUCHE HAUT"],
            [[0.0506, -0.201, 2.0847, 0.0107, -0.0844, 0.9112], [-0.1381, -0.1304], "un peu en haut"],
            [[0.158, -0.8007, 2.0847, 0.0982, -0.0599, 1.0], [-0.1396, -0.1304], "retour vers depart"],
            [[0.6642, -0.4893, 2.0801, 0.0951, -0.0721, 0.0404], [-0.0966, -0.1028], "goSide"],
        ]
        
        step_idx = 0
        move_step = 0.05 

        print("\n============================================")
        print("      CONTROLE DYNAMIQUE (POSITION REELLE)  ")
        print("============================================")
        print(" N : Scenario (Bras) | PAVE : Ajuster Bras")
        print(" +/- : Ajuster Poignet | Y : Export pos")
        print(" IJKL : Ajuster Tete | ZSQD/AE : Base")
        print("============================================\n")

        while True:
            key = get_key().lower()
            
            # --- LOGIQUE SCENARIO ---
            if key == 'n':
                if step_idx < len(scenario):
                    a_vals, h_vals, msg = scenario[step_idx]
                    motion.post.setAngles(arm_names, a_vals, 0.1)
                    print("[>] Etape %d : %s" % (step_idx + 1, msg))
                    step_idx += 1

            elif key == 'r':
                step_idx = 0
                print("\n[!] Scenario remis a zero.")

            # --- AJUSTEMENT DYNAMIQUE (BRAS ET POIGNET) ---
            # Ajout de '+' et '-' dans la liste des touches acceptées
            elif key in ['8', '2', '4', '6', '7', '9', '1', '3', '5', '0', '+', '-']:
                cur_arm = motion.getAngles("RArm", True)
                arm_dict = dict(zip(arm_names, cur_arm))
                
                if key == '8': arm_dict["RShoulderPitch"] -= move_step
                elif key == '2': arm_dict["RShoulderPitch"] += move_step
                elif key == '6': arm_dict["RShoulderRoll"] -= move_step
                elif key == '4': arm_dict["RShoulderRoll"] += move_step
                elif key == '7': arm_dict["RElbowRoll"] += move_step
                elif key == '9': arm_dict["RElbowRoll"] -= move_step
                elif key == '1': arm_dict["RElbowYaw"] -= move_step
                elif key == '3': arm_dict["RElbowYaw"] += move_step
                elif key == '5': arm_dict["RHand"] = 1.0
                elif key == '0': arm_dict["RHand"] = 0.0
                elif key == '+': arm_dict["RWristYaw"] += move_step  # Rotation du poignet
                elif key == '-': arm_dict["RWristYaw"] -= move_step  # Rotation du poignet
                
                # Application immediate (securite incluse pour RShoulderRoll)
                arm_dict["RShoulderRoll"] = max(-1.5, min(-0.1, arm_dict["RShoulderRoll"]))
                motion.post.setAngles(arm_names, [arm_dict[name] for name in arm_names], 0.1)

            # --- AJUSTEMENT DYNAMIQUE (TETE) ---
            elif key in ['i', 'k', 'j', 'l']:
                cur_head = motion.getAngles("Head", True)
                head_dict = dict(zip(head_names, cur_head))

                if key == 'i': head_dict["HeadPitch"] -= move_step
                elif key == 'k': head_dict["HeadPitch"] += move_step
                elif key == 'j': head_dict["HeadYaw"] += move_step
                elif key == 'l': head_dict["HeadYaw"] -= move_step
                
                head_dict["HeadPitch"] = max(-0.6, min(0.6, head_dict["HeadPitch"]))
                motion.post.setAngles(head_names, [head_dict[name] for name in head_names], 0.1)

            # --- ROUES ---
            elif key == 'z': motion.post.moveTo(0.2, 0.0, 0.0)
            elif key == 's': motion.post.moveTo(-0.2, 0.0, 0.0)
            elif key == 'q': motion.post.moveTo(0.0, 0.1, 0.0)
            elif key == 'd': motion.post.moveTo(0.0, -0.1, 0.0)
            elif key == 'a': motion.post.moveTo(0.0, 0.0, 0.15) 
            elif key == 'e': motion.post.moveTo(0.0, 0.0, -0.15) 

            # --- EXPORT ---
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