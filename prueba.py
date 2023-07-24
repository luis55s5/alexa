#------------------Importamos librerias-------------------
import cv2	#OpenCV
import Informacion_Manos as im	#importamos la informacion de las manos 
import os	#para pasar de carpeta
import numpy as np
from keras_preprocessing.image import load_img, img_to_array	#para pasar la imagen a una matriz
from keras.models import load_model		#para cargar los modelos 
import speech_recognition as sr
import pyttsx3
import json
import time

start_time = time()
engine = pyttsx3.init()

# name of the virtual assistant
name = 'carlos'
attemts = 0

with open('keys.json') as json_file:
    keys = json.load(json_file)
    

#------------------Ubicacion del modelo y los pesos---------------------------
modelo = '/home/pi/Desktop/pruebas/ModeloObjetos.h5'
peso = '/home/pi/Desktop/pruebas/pesosObjetos.h5'

# colors
green_color = "\033[1;32;40m"
red_color = "\033[1;31;40m"
normal_color = "\033[0;37;40m"

#---------------------Cargamos el modelo-------------------------
cnn = load_model(modelo)	#Cargamos el modelo
cnn.load_weights(peso)	#Cargamos los pesos

#-------------------Cargamos los nombres de las carpetas------------------
direccion = '/home/pi/Desktop/pruebas/fotos/Validacion'
dire_img = os.listdir(direccion)
print("Nombres", dire_img)

#------------------Declaracion de las variables-------------------------
anchocam, altocam = 640, 480

#----------------Lectura de camara----------------------------------

cap = cv2.VideoCapture(0)
cap.set(3,anchocam)		#Definiremos un ancho y un alto definido para siempre
cap.set(4,altocam)

#-----------------------------Declaramos el detector---------------------------------
detector = im.detectormanos(maxManos=1, Confdeteccion=0.7)	#Ya que solo vamos a utilizar una mano

# get voices and set the first of them
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[20].id)

# editing default configuration
engine.setProperty('rate', 178)
engine.setProperty('volume', 0.7)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_audio():
    r = sr.Recognizer()
    status = False

    with sr.Microphone() as source:
        print(f"{green_color}({attemts}) Escuchando...{normal_color}")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        rec = ""
        
        try:
            rec = r.recognize_google(audio, language='es-ES').lower()
            
            if name in rec:
                rec = rec.replace(f"{name} ", "").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
                status = True
            else:
                speak(f"Vuelve a intentarlo, no reconozco: {rec}")
        except:
            pass
    return {'text':rec, 'status':status}

    
def openCam():
    while True:
        rec_json = get_audio()

        rec = rec_json['text']
        status = rec_json['status']
       

        #-------------------------Vamos a encontrar los puntos de la mano-----------------------------
        ret, frame = cap.read()		#Hacemos la lectura de la camara
        mano = detector.encontrarmanos(frame)	#Encontramos las manos
        lista, bbox = detector.encontrarposicion(frame)	#Mostramos las posiciones
                                                        #bbox son las coordenadas del rectangulo que rode nuestra mano
        if len(lista) != 0:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            data = frame[y1:y2, x1:x2]	#Volvemos a almacenar los pixeles que estan en el recuadro
            obje = cv2.resize(data, (200, 200), interpolation=cv2.INTER_CUBIC)	#Redimensionamos las fotos
            x = img_to_array(obje)	#Convertiremos la imagen a una matriz
            x = np.expand_dims(x, axis=0)	#Agregamos nuevos eje
            vector = cnn.predict(x)		#Va a ser un arreglo de 2 dimensiones, donde va a poner 1 en la calse que crea correcta
            resultado = vector[0]	#[1,0,0] | [0,1,0] | [0,0,1]
            respuesta = np.argmax(resultado)	#Nos entrega el indice del valor mas alto
            if respuesta == 0:
                print(resultado)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, speak('{}').format(dire_img[0]), (x1, y1 - 100), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            elif respuesta == 1:
                print(resultado)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, speak('{}').format(dire_img[1]), (x1, y1 - 100), 1, 2.5, (255, 255, 0), 3, cv2.LINE_AA)
            elif respuesta == 2:
                print(resultado)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, speak('{}').format(dire_img[2]), (x1, y1 - 100), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'OBJETO DESCONOCIDO', (x1, y1 - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
                
        cv2.imshow("Clasificador", frame)
#         k = cv2.waitKey(1)
        
        #time.sleep(300)
        
       
        if status:
            if 'salir' in rec:
                speak('saliendo')
                break
    cap.realease()
    cv2.destroyAllWindows()
openCam()
