# -*- coding: utf-8 -*-
import time
from naoqi import ALProxy

def execute_symmetrical_cinema(robot_ip="127.0.0.1", port=9559):
    try:
        print("Connexion aux services...")
        motion = ALProxy("ALMotion", robot_ip, port)
        posture = ALProxy("ALRobotPosture", robot_ip, port)
        tts = ALProxy("ALTextToSpeech", robot_ip, port)

        # --- ÉTAPE CRITIQUE : LE RÉVEIL ---
        print("Reveil du robot (Activation des moteurs)...")
        motion.wakeUp() 
        
        print("Mise en position StandInit...")
        posture.goToPosture("StandInit", 0.5)
        
        # Securite : on force la raideur au maximum
        motion.setStiffnesses("Body", 1.0)
        
        # Regard fixe
        motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, 0.0], 0.2)
        time.sleep(1.0)

        # 2. POSE : ABSOLUTE CINEMA
        print("Preparation de la pose...")
        names = [
            "RShoulderPitch", "LShoulderPitch", 
            "RShoulderRoll",  "LShoulderRoll",  
            "RElbowRoll",     "LElbowRoll",     
            "RWristYaw",      "LWristYaw",      
            "RHand",          "LHand"           
        ]

        angles = [
            0.0, 0.0,     # Hauteur
            -1.2, 1.2,    # Ouverture
            1.5, -1.5,    # Coudes
            1.5, -1.5,    # Poignets
            1.0, 1.0      # Mains
        ]

        times = [0.6] * len(names)

        print("Mouvement en cours !")
        motion.angleInterpolation(names, angles, times, True)
        
        tts.setLanguage("English")
        tts.say("Absolute Cinema.")

        time.sleep(5.0)
        print("Pose terminee.")

    except Exception as e:
        print("ERREUR : {}".format(e))

if __name__ == "__main__":
    # Si tu lances le script depuis ton PC, remplace 127.0.0.1 par l'IP du robot
    execute_symmetrical_cinema("127.0.0.1", 9559)