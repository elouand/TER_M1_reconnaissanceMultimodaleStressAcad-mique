# -*- coding: utf-8 -*-
import socket
import json
import time
from naoqi import ALProxy, ALBroker

PORT_RETOUR = 5007

# 1. CRÉER LE BROKER EN PREMIER
try:
    myBroker = ALBroker("myBroker", "0.0.0.0", 0, "127.0.0.1", 9559)
except Exception as e:
    print "Erreur Broker :", e
    exit(1)

# 2. CRÉER LES PROXYS ENSUITE
try:
    tts = ALProxy("ALTextToSpeech")
    motion = ALProxy("ALMotion")
    animation = ALProxy("ALAnimationPlayer") 
except Exception as e:
    print "Erreur lors de la creation des proxys :", e

def get_tablet_proxy():
    try:
        return ALProxy("ALTabletService")
    except Exception as e:
        print "Erreur proxy Tablette :", e
        return None

tablet = get_tablet_proxy()

if tablet:
    print "Connexion reussie a la tablette !"
    tablet.showWebview("about:blank")
    time.sleep(1)
else:
    print "ERREUR : Tablette indisponible."

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT_RETOUR))

def reaction_robot():
    """ Fermeture de la main (0.0 = fermé, 1.0 = ouvert) """
    print "EXECUTION MOUVEMENT : Fermeture des mains"
    try:
        # Vitesse de 0.2 pour un mouvement net mais pas trop brusque
        motion.setAngles(["RHand", "LHand"], [0.0, 0.0], 0.2)
    except Exception as e:
        print "Erreur mouvement main :", e

print "En attente des donnees sur le port 5007..."

try:
    while True:
        data, addr = sock.recvfrom(1024)
        
        try:
            info = json.loads(data)
            status = info.get("status")

            # --- ETAPE 1 : RÉACTION AU SIGNAL ---
            if info.get("move") == True:
                reaction_robot()

            # --- AFFICHAGE TABLETTE (Correction du 'd') ---
            if status == "ok" and tablet:
                v, a = info["v"], info["a"]
                cmd = "document.body.style.backgroundColor = 'green'; "
                cmd += "document.body.innerHTML = '<div style=\"color:white; font-size:60px; text-align:center; margin-top:150px;\">"
                cmd += "V: %.2f <br> A: %.2f</div>';" % (v, a)
                tablet.executeJS(cmd)

            elif status == "none" and tablet:
                cmd = "document.body.style.backgroundColor = 'red'; "
                cmd += "document.body.innerHTML = '<h1 style=\"color:white; font-size:60px; text-align:center; margin-top:150px;\">RECHERCHE...</h1>';"
                tablet.executeJS(cmd)

        except Exception as e:
            print "Erreur traitement JSON :", e

except KeyboardInterrupt:
    print "Arret."
    if myBroker: myBroker.shutdown()
finally:
    sock.close()