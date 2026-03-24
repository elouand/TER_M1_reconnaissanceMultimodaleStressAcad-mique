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
    exit(1) # Si le broker échoue, on ne peut rien faire

# 2. CRÉER LES PROXYS ENSUITE
try:
    tts = ALProxy("ALTextToSpeech")
    motion = ALProxy("ALMotion")
    # Ajout du proxy pour les animations (Etape 2)
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
    """ Fonction de test pour l'étape 1 """
    print "EXECUTION MOUVEMENT"
    # Utilisation de .post pour ne pas bloquer la boucle UDP
    tts.post.say("mouvement")
    # Exemple d'animation pour l'étape 2 (à tester si tts fonctionne)
    # animation.post.run("animations/Stand/Gestures/Hey_1")

print "En attente des donnees sur le port 5007..."

try:
    while True:
        data, addr = sock.recvfrom(1024)
        # On peut commenter le print suivant si ça flood trop tes logs
        # print "Paquet recu de %s" % str(addr)
        
        try:
            info = json.loads(data)
            status = info.get("status")

            # --- ETAPE 1 : RÉACTION AU SIGNAL ---
            if info.get("move") == True:
                reaction_robot()

            # --- AFFICHAGE TABLETTE ---
            if status == "ok" and tablet:
                v, a, d = info["v"], info["a"], info["d"]
                cmd = "document.body.style.backgroundColor = 'green'; "
                cmd += "document.body.innerHTML = '<div style=\"color:white; font-size:50px; text-align:center; margin-top:100px;\">"
                cmd += "V: %.2f <br> A: %.2f <br> D: %.2f</div>';" % (v, a, d)
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
