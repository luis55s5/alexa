#------------------------Importamos librerias--------------------------
import math	
import cv2
import mediapipe as mp		#Ayuda a colocar los 21 puntos de las manos


#--------------------------Creamos una clase----------------------------
class detectormanos():
    #-----------------------Iniciamos los parametros de la deteccion-------
    def __init__(self, mode=False, maxManos= 2, Confdeteccion= 0.5, modelComplexity=1, Confsegui= 0.5):
        self.mode = mode		#creamos el objeto y él tendrá su propia variable
        self.maxManos = maxManos		#Lo mismo haremos con todos los objetos
        self.Confdeteccion = Confdeteccion
        self.modelComplex = modelComplexity
        self.Confsegui = Confsegui
        
        #----------------Creamos los objetos que detectarán las manos y los dibujaran---------
        self.mpmanos = mp.solutions.hands
        self.manos = self.mpmanos.Hands(self.mode, self.maxManos, self.modelComplex, self.Confdeteccion, self.Confsegui)
        self.dibujo = mp.solutions.drawing_utils	#Dibujamos las conexiones entre los puntos de las manos
        self.tip = [4,8,12,16,20]
        
    #-------------------Funcion para encontrar las manos--------------------
    def encontrarmanos(self, frame, dibujar = True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #pasamos de BGR a RGB la imagen
        self.resultados = self.manos.process(imgcolor)
        
        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame, mano, self.mpmanos.HAND_CONNECTIONS) #Dibujamos las conexiones de los puntos de la mano
        return frame
    #-----------------------Función para encontrar la posición----------------------------------
    #nos entrega los 21 pintos de interes y las cordenadas del rectangulo donde se encuentra nuestra mano
    def encontrarposicion(self, frame, ManoNum = 0, dibujar = True):
        xlista = []
        ylista = []
        bbox = []
        
        self.lista = []
        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[ManoNum]
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape #Extraemos las dimensiones de los fps
                cx, cy = int(lm.x * ancho), int(lm.y * alto) # Convertimos la informacion en pixeles
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id, cx, cy])
                if dibujar:
                    cv2.circle(frame,(cx, cy), 5, (0, 0, 0), cv2.FILLED) #Dibujamos un circulo
            
            xmin, xmax = min(xlista), max(xlista)
            ymin, ymax = min(ylista), max(ylista)
            bbox = xmin, ymin, xmax, ymax		# Enn bbox os entrega las coordenadas de nuestro rectangulo 
            if dibujar:
                cv2.rectangle(frame,(xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0,255,0),2)
        return self.lista, bbox		#lista nos entrega las coordenadas de los 21 puntos de interes
    
    #---------------------------Función oara detectar la distancia entre los dedos------------------
    def distancia(self, p1, p2, frame, dibujar = True, r = 15, t = 3):
        x1, y1 = self.lista[p1][1:]
        y1, y2 = self.lista[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 +y2) // 2
        if dibujar:
            cv2.line(frame, (x1,y1), (x2,y2), (0,0,255),t)
            cv2.circle(frame, (x1,y1), r, (0,0,255), cv2.FILLED)
            cv2.circle(frame, (x2,y2), r, (0,0,255), cv2.FILLED)
            cv2.circle(frame, (cx,cy), r, (0,0,255), cv2.FILLED)
        lenght = math.hypot(x2-x1, y2-y1)
        
        return lenght, frame, [x1, y1, x2, y2, cx, cy]
    
#---------------------------------------Funcion principal----------------------------
def main():
    ptiempo = 0
    ctiempo = 0
    
    #---------------------------------Leemos la camara web------------------------
    cap = cv2.VideoCapture(0)
    
    #---------------------------------Creamos el objeto-----------------------
    detector = detectormanos()
    
    #---------------------------------Realizamos la deteccion de manos-----------
    while True:
        ret, frame = cap.read()
        
        # Una vez que obtengamos al imagen enciaremos
        frame = detector.encontrarmanos(frame)
        lista, bbox = detector.encontrarposicion(frame)
        if len(lista) !=0:
            print(lista[4])
        
        cv2.imshow("Manos", frame)
        k = cv2.waitKey(1)  
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()