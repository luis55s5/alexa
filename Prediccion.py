#------------------Importamos librerias-------------------
import cv2	#OpenCV
import Informacion_Manos as im	#importamos la informacion de las manos 
import os	#para pasar de carpeta
import numpy as np
from keras_preprocessing.image import load_img, img_to_array	#para pasar la imagen a una matriz
from keras.models import load_model		#para cargar los modelos 

#------------------Ubicacion del modelo y los pesos---------------------------
modelo = '/home/pi/Desktop/pruebas/ModeloObjetos.h5'
peso = '/home/pi/Desktop/pruebas/pesosObjetos.h5'

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

while True:
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
            cv2.putText(frame, '{}'.format(dire_img[0]), (x1, y1 - 100), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
        elif respuesta == 1:
            print(resultado)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, '{}'.format(dire_img[1]), (x1, y1 - 100), 1, 2.5, (255, 255, 0), 3, cv2.LINE_AA)
        elif respuesta == 2:
            print(resultado)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, '{}'.format(dire_img[2]), (x1, y1 - 100), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'OBJETO DESCONOCIDO', (x1, y1 - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
            
    cv2.imshow("Clasificador", frame)
    k = cv2.waitKey(1)
    if k == 27:		#Salimos con ESC.
        break
cap.realease()
cv2.destroyAllWindows()