# -*- coding: utf-8 -*-
import socket
import json
import time
import os
import urllib
import threading
import SimpleHTTPServer
import SocketServer
from naoqi import ALProxy, ALBroker

PORT_RETOUR = 5007

# --- CACHE IMAGE & SERVEUR WEB LOCAL ---
# C'est ici que Pepper va ranger les images venant du PC
CACHE_DIR = "/home/nao/img_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Création d'un serveur qui tolère les redémarrages de script
class ReusableTCPServer(SocketServer.TCPServer):
    allow_reuse_address = True

def run_local_server():
    os.chdir(CACHE_DIR)
    Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = ReusableTCPServer(("0.0.0.0", 8080), Handler)
    print "🌐 Serveur d'images interne demarre sur le port 8080"
    httpd.serve_forever()

# Démarrer le serveur web interne en arrière-plan
t = threading.Thread(target=run_local_server)
t.daemon = True
t.start()

# 1. CRÉER LE BROKER
try:
    myBroker = ALBroker("myBroker", "0.0.0.0", 0, "127.0.0.1", 9559)
except Exception as e:
    print "Erreur Broker :", e
    exit(1)

# 2. CRÉER LES PROXYS
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
        motion.setAngles(["RHand", "LHand"], [0.0, 0.0], 0.2)
    except Exception as e:
        print "Erreur mouvement main :", e

print "En attente des donnees sur le port 5007..."

try:
    while True:
        data, addr = sock.recvfrom(1024)
        pc_ip = addr[0] # L'adresse IP de ton PC (169.254...)

        try:
            info = json.loads(data)
            status = info.get("status")
            image_name = info.get("image", "")

            # --- ETAPE 1 : RÉACTION AU SIGNAL ---
            if info.get("move") == True:
                reaction_robot()

            # --- AFFICHAGE TABLETTE ---
            if tablet:
                if status == "ok":
                    # Style CSS: Image centrée sur fond noir, SANS overlay vert
                    html = "<style>"
                    html += "body { margin: 0; background-color: black; overflow: hidden; display: flex; justify-content: center; align-items: center; height: 100vh; }"
                    html += "img { max-width: 100%; max-height: 100%; object-fit: contain; }"
                    html += "</style>"
                    
                    if image_name:
                        # 1. Le Cerveau télécharge l'image depuis ton PC
                        img_url_pc = "http://%s:8000/images/%s" % (pc_ip, image_name)
                        local_path = os.path.join(CACHE_DIR, image_name)
                        
                        if not os.path.exists(local_path):
                            print "⏬ Telechargement en cache de :", image_name
                            try:
                                urllib.urlretrieve(img_url_pc, local_path)
                            except Exception as e:
                                print "Erreur DL:", e

                        # 2. La tablette affiche l'image depuis le mini-serveur interne
                        img_url_tablet = "http://198.18.0.1:8080/%s" % image_name
                        html += "<img src='%s'>" % img_url_tablet
                    else:
                        html += "<h1 style='color:white; font-size:60px;'>EN PAUSE<br>Appuyez sur R</h1>"
                    
                    # On crée la commande et on force l'encodage pour la tablette
                    cmd = "document.body.innerHTML = \"%s\";" % html.replace('"', '\\"')
                    tablet.executeJS(cmd.encode('utf-8'))

                elif status == "none":
                    cmd = "document.body.style.backgroundColor = 'red'; "
                    cmd += "document.body.innerHTML = '<h1 style=\"color:white; font-size:60px; text-align:center; margin-top:200px;\">RECHERCHE VISAGE...</h1>';"
                    tablet.executeJS(cmd.encode('utf-8'))

        except Exception as e:
            print "Erreur traitement JSON :", e

except KeyboardInterrupt:
    print "Arret."
    if myBroker: myBroker.shutdown()
finally:
    sock.close()
