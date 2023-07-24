#----------------------------Importamos librerias------------------------
import tensorflow.keras.optimizers

#----------------------------Creaar modelo y entrenamiento-----------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator #Nos ayuda a preprocesar las imagenes que le entreguemos
from tensorflow.keras import optimizers		#Opimizador con el que se va a entrenar el modelo
from tensorflow.keras.models import Sequential		#Nos permite hacer redes neuronales secuenciales
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation #Libreria de activacion
from tensorflow.keras.layers import Convolution2D, MaxPooling2D		#Capas para hacer las convoluciones
from tensorflow.keras import backend as K		#Si hay una sesion de keras, lo cerramos para tener todo limpio

K.clear_session()		#Limpiamos todo

datos_entrenamiento = '/home/pi/Desktop/Clasificador de objetos/fotos/Entrenamiento'
datos_validacion = '/home/pi/Desktop/Clasificador de objetos/fotos/Validacion'

#-------------------------Parametros-----------------------------------
iteraciones = 20	#Numero de iteraciones para ajustar nuestro modelo
altura, longitud = 200, 200		#Tamaño de las imagenes de entrenamiento
batch_size = 1		#Numero de imagenes que vamos a enviar
pasos = 300/1		#Numero de veces que se va a procesar la informacion en cada iteracion
pasos_validacion = 300/1	#Despues de cada iteracion, validamos lo anterior
filtrosconv1 = 32
filtrosconv2 = 64		#Numero de filtros que vamos a aplicar en cada convolucion
filtrosconv3 = 128
tam_filtro1 =(4,4)
tam_filtro2 =(3,3)		#Tamaños de los filtros 1, 2 y 3
tam_filtro3 =(2,2)
tam_pool = (2,2)	#Tamaño del filtro en max pooling
clases = 3		#3 Objetos (mouse, celular, multimetro) cantidad de objetos que agregamos 
lr = 0.0005		#Ajustes de al red neuronal para acercarse a una solucion optima

#---------------Pre-procesamiento de las imagenes-----------------------------
preprocesamiento_entre = ImageDataGenerator(
    rescale = 1./255,		#Pasar los objetos de 0 a 255 | 0 a 1
    shear_range = 0.3,		#Generar nuestras imagenes incluidas para un mejor entrenamiento
    zoom_range = 0.3,		#Genera imagenes con zoom para un mejor entrenamiento
    horizontal_flip = True	#Invierte las imagenes para mejor entrenamiento
)

preprocesamiento_vali = ImageDataGenerator(
    rescale = 1./255		#Preporcesamineto de la validación que no queremos que se modifique tanto ya que en el paso anterior ya lo hizimos
)

imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,		#Va a tomar las fotos que ya almacenamos
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical',		#Clasificacion categorica = por clases
)

imagen_validacion = preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

#-----------------------Creamos la red neuronal convolucional (CNN)------------------------
cnn= Sequential()		#Red neuronal secuencial

#----------------------Agregamos filtros con el fin de volver nuestra imagen muy profunda pero pequeña-----------
cnn.add(Convolution2D(filtrosconv1, tam_filtro1, padding = 'same', input_shape=(altura,longitud,3), activation = 'relu'))	#Agregamos la primera capa 
        #Es una convolucion y realizamos config
cnn.add(MaxPooling2D(pool_size=tam_pool))	#Despues de la primera capa vamos a tener una capa de max pooling y asignamos el tamaño
                                            #MaxPooling es la extraccion de caracteristicas
cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding = 'same', activation = 'relu'))	#Agregamos nueva capa
cnn.add(MaxPooling2D(pool_size=tam_pool))

#---------------Nueva capa-------------------
cnn.add(Convolution2D(filtrosconv3, tam_filtro3, padding = 'same', activation = 'relu'))		#Agregamos nueva capa
cnn.add(MaxPooling2D(pool_size=tam_pool))

#------------Ahora vamos a convertir esa imagen profunda a una plana, para tener 1 dimension con toda la informacion-------------
cnn.add(Flatten())	#Aplanemos la imagen
cnn.add(Dense(384,  activation='relu'))	#Asignamos 384 neuronas
cnn.add(Dropout(0.5))	#Apagamos el 50% de las neuronas en la funcion anterior para no sobre-ajustar la red
cnn.add(Dense(clases, activation='softmax'))	#Es nuestra ultima capa, es la que nos dice la probabilidad de que sea alguna de los objetos entrenados

#---------------Agregamos parametros para optimizar el modelo----------------------------
#Durante el entrenamiento tenga una autoevaluacion, que es optimice en Adam, y la metrica sera accuracy
optimizar = tensorflow.keras.optimizers.Adam(learning_rate = lr)
cnn.compile(loss = 'categorical_crossentropy', optimizer=optimizar, metrics=['accuracy'])	#Durante el entrenamiento que tenga un auto-evaluacion (categorical_crossentropy), el optimizador es Adam y la metrica es accuracy

#-----------------Entrenaremos nuestra red---------------------------
cnn.fit(imagen_entreno, steps_per_epoch=pasos, epochs=iteraciones, validation_data=imagen_validacion, validation_steps=pasos_validacion)

#---------------Guardamos el modelo---------------------------------
cnn.save('ModeloObjetos.h5')
#-----------------Guardamos los peso del modelo------------------
cnn.save_weights('pesosObjetos.h5')