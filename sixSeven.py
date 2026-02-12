# -*- coding: utf-8 -*-
import time
from naoqi import ALProxy

def faire_sixSeven(robot_ip="127.0.0.1", port=9559):
    try:
        # Initialisation des services
        motion = ALProxy("ALMotion", robot_ip, port)
        tablet = ALProxy("ALTabletService", robot_ip, port)
        posture = ALProxy("ALRobotPosture", robot_ip, port)
        tts = ALProxy("ALTextToSpeech", robot_ip, port)
        photo_capture = ALProxy("ALPhotoCapture", robot_ip, port)

        # 1. POSTURE ET VERROUILLAGE TÊTE
        posture.goToPosture("StandInit", 0.5)
        
        print("Fixation du regard...")
        motion.setStiffnesses("Head", 1.0)
        motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, 0.0], 0.2)
        
        # Pause de stabilisation pour une photo nette
        time.sleep(2.0)

        # --- SÉQUENCE PHOTO AVEC NOM UNIQUE (ANTI-CACHE) ---
        tts.setLanguage("French")
        tts.say("Attention pour la nouvelle photo.")
        
        # On cree un nom unique avec le temps actuel (ex: photo_1707745000)
        timestamp = str(int(time.time()))
        photo_name = "photo_" + timestamp
        photo_folder = "/home/nao/recordings/cameras"
        
        print("Prise de la photo : " + photo_name)
        photo_capture.setResolution(2) # VGA
        photo_capture.takePicture(photo_folder, photo_name)
        
        # On attend une demi-seconde que le fichier soit bien ecrit sur le disque
        time.sleep(0.5)
        
        # Affichage via le serveur Python (port 8000)
        photo_url = "http://198.18.0.1:8000/" + photo_name + ".jpg"
        print("Affichage de l'URL : " + photo_url)
        tablet.showImage(photo_url)
        
        time.sleep(1.0) 

        # 2. CONFIGURATION DE LA DANSE (COUDES DYNAMIQUES)
        names = [
            "RShoulderPitch", "LShoulderPitch", 
            "RElbowRoll", "LElbowRoll", 
            "RWristYaw", "LWristYaw"
        ]
        
        # Mouvements : Epaule monte -> Coude plie / Epaule descend -> Coude tend
        pos1 = [0.6, 1.4, 1.5, -0.2, 1.5, -1.5]
        pos2 = [1.4, 0.6, 0.2, -1.5, 1.5, -1.5]

        print("Lancement Six Seven...")
        tts.setLanguage("English")
        
        # La tete reste fixe pour le style robotique
        motion.setStiffnesses("Head", 1.0)

        for i in range(5):
            tts.post.say("Six Seven")
            motion.setAngles(names, pos1, 0.35)
            time.sleep(0.7)
            motion.setAngles(names, pos2, 0.35)
            time.sleep(0.7)

        # 3. FIN DE L'ANIMATION
        posture.goToPosture("StandInit", 0.5)
        
        # On relache la raideur de la tete
        motion.setStiffnesses("Head", 0.0)
        
        print("Termine. Nouvelle photo affichee.")

    except Exception as e:
        print("Erreur : " + str(e))

if __name__ == "__main__":
    # RAPPEL : Le serveur Python doit tourner dans ~/recordings/cameras
    # cd ~/recordings/cameras && python -m SimpleHTTPServer 8000 &
    faire_sixSeven()