#----------------------------Importamos librerias-------------------------
import cv2
import Informacion_Manos as im # Programa que contiene la deteccion y seguimiento de las manos
import os	# para movernos entre archivos

#------------------------Creamos la carpeta donde almacenamos le entrenamineto-------------------
nombre = 'multimetro'		#Asignamos el nombre de la carpeta 
direccion = '/home/pi/Desktop/Clasificador de objetos/fotos/Validacion' #donde guardaremos las fotos para entrenamiendo y validacion
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print('Carpeta creada: ' ,carpeta) # Si no hemos creado la carpeta este lo creará por si solo
    os.makedirs(carpeta)
    
#------ Asignamos un contador para el nombre de las fotos--------
cont = 0

#-----------------------------Declaración de variables--------------------------
anchocam, altocam = 640, 480 #determinar el ancho y el alto de la ventana de la camara

#----------------------------Lectura de la camara------------------------------
cap = cv2.VideoCapture(0)
cap.set(3,anchocam)		#Definí un ancho y un alto para siempre
cap.set(4,altocam)

#---------------------------Declaramos el detector ------------------------------
detector = im.detectormanos(maxManos=1, Confdeteccion=0.7)	#colocamos la cantidad de manos que se quiere detectar en este caso será 1 y con una confianza de 0.7

while True:
    #-------------------------Vamos a encontrar los puntos de la mano------------------
    ret, frame = cap.read() # frame son los fotogramas que toma la camara 
    mano = detector.encontrarmanos(frame)	#Encontramos las manos 
    lista, bbox = detector.encontrarposicion(frame)	#Mostramos las posiciones
    if len(bbox) !=0:	#si bbox es diferente de 0 extraemos las coordenadas 
        x1 = bbox[0]	#extraemos la informacion en la posicion 0,1,2 y 3
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        data = frame[y1:y2, x1:x2]	#alacenamos los pixeles que conforman el recuadro de nuestra mano
        obje = cv2.resize(data, (200, 200), interpolation=cv2.INTER_CUBIC)	#Redimensionamos las fotos para le entrenamiento de la red neuronal
        cv2.imwrite(carpeta + "/Objeto_{}.jpg".format(cont), obje) #carpeta almacenamos las imagenes de 200x200 fps
        cont = cont + 1		#El contador esto aumuntará 1 en 1 de 0 a 299
        
        
    cv2.imshow("Base Datos", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:		#Si le damos la tecla Esc o acaban las 300 imagenes se cerrará la ventana
        break
cap.release()
cv2.destroyAllWindows()