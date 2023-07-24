import cv2
import numpy as np
import serial

COM = '/dev/ttyUSB0'
BAUD = 9600
ser = serial.Serial(COM, BAUD)

cap = cv2.VideoCapture(0)
rojoBajo1 = np.array([0, 100, 20], np.uint8)
rojoAlto1 = np.array([5, 255, 255], np.uint8)

rojoBajo2 = np.array([175, 100, 20], np.uint8)
rojoAlto2 = np.array([179, 255, 255], np.uint8)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara1 = cv2.inRange(frameHSV, rojoBajo1, rojoAlto1)
        mascara2 = cv2.inRange(frameHSV, rojoBajo2, rojoAlto2)
        mascara = cv2.add(mascara1, mascara2)
        
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #output = frame.copy()
        cv2.drawContours(frame, contornos, -1, (0, 0, 255), 4)
        
        for c in contornos:
            area = cv2.contourArea(c)
            if area > 6000:
                M = cv2.moments(c)
                if M["m00"] == 0:
                    M["m00"] = 1
                x = int(M["m10"] / M["m00"])
                y = int(M['m01'] / M['m00'])
                cv2.circle(frame, (x, y), 7, (255, 0, 0), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
                nuevoContorno = cv2.convexHull(c)
                cv2.drawContours(frame, [nuevoContorno], 0, (0, 0, 255), 3)
                
                if x < 52:
                    print("Mover a la izquierda 100%")
                    ser.write(b"izq1\n")
                elif 105 > x >= 52:
                    print("Mover a la izquierda 80%")
                    ser.write(b"izq2\n")
                elif 158 > x >= 105:
                    print("Mover a la izquierda 60%")
                    ser.write(b"izq3\n")
                elif 210 > x >= 158:
                    print("Mover a la izquierda 40%")
                    ser.write(b"izq4\n")
                elif 263 > x >= 210:
                    print("Mover a la izquierda 20%")
                    ser.write(b"izq5\n")
                #Mover al centro
                elif 263 <= x < 369:
                    print("Mover al centro")
                    ser.write(b"ctr\n")
                elif 369 <= x < 421:
                    print("Mover a la derecha 20%")
                    ser.write(b"der5\n")
                elif 421 <= x < 474:
                    print("Mover a la derecha 40%")
                    ser.write(b"der4\n")
                elif 474 <= x < 527:
                    print("Mover a la derecha 60%")
                    ser.write(b"der3\n")
                elif 527 <= x < 575:
                    print("Mover a la derecha 80%")
                    ser.write(b"der2\n")
                elif x >= 575:
                    print("Mover a la derecha 100%")
                    ser.write(b"der1\n")
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            ser.close()
            break
cap.realease()
cv2.destroyAllWindows()
                