import cv2
import os
import numpy as np
import time

def obtenerModelo(method,facesData,labels,muestra):
	if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
	if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Entrenando el reconocedor de rostros
	print("Entrenando ( "+method+" )...")
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+method+muestra+" ): ", tiempoEntrenamiento)

	# Almacenando el modelo obtenido
	emotion_recognizer.write("modelo"+method+muestra+".xml")

#dataPath = 'C:/Users/Paula/Downloads/INSTALADORES TESIS/Output/7030/Train' #Cambia a la ruta donde hayas almacenado Data
dataPath = 'C:/Users/Paula/Downloads/INSTALADORES TESIS/Output/7525/Train' #Cambia a la ruta donde hayas almacenado Data
#dataPath = 'C:/Users/Paula/Downloads/INSTALADORES TESIS/Output/8020/Train' #Cambia a la ruta donde hayas almacenado Data

emotionsList = os.listdir(dataPath)
print('Lista de personas: ', emotionsList)

labels = []
facesData = []
label = 0

for nameDir in emotionsList:
	emotionsPath = dataPath + '/' + nameDir

	for fileName in os.listdir(emotionsPath):
		#print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(emotionsPath+'/'+fileName,0))
		#image = cv2.imread(emotionsPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1
print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))
print('Número de etiquetas 2: ',np.count_nonzero(np.array(labels)==2))
print('Número de etiquetas 3: ',np.count_nonzero(np.array(labels)==3))
print('Número de etiquetas 4: ',np.count_nonzero(np.array(labels)==4))
print('Número de etiquetas 5: ',np.count_nonzero(np.array(labels)==5))

# ----------- Muestra usados para el entrenamiento y lectura del modelo ----------
#muestra = '7030'
muestra = '7525'
#muestra = '8020'

#obtenerModelo('LBPH',facesData,labels,muestra)
obtenerModelo('EigenFaces',facesData,labels,muestra)
#obtenerModelo('FisherFaces',facesData,labels,muestra)