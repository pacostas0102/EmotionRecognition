from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn import metrics
# Rendimiento del clasificador visual
# load libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import cv2
import os
import imutils
import numpy as np

#---------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------
# ------------------- Métodos usados para el entrenamiento y lectura del modelo ----------------
#method = 'EigenFaces'
#method = 'FisherFaces'
method = 'LBPH'
# --------------------- Muestra usados para el entrenamiento y lectura del modelo --------------
#muestra = '7030'
#muestra = '7525'
muestra = '8020'

print(method)

if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+muestra+'.xml')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# -------------------------------------------------------------------------------------------------------
 
# Cargar datos -----------------------------------------------------------------------------------------
#dataPath = 'C:/Users/Paula/Downloads/INSTALADORES TESIS/EmotionRecognition/Output/7030/Validation' #Cambia a la ruta donde hayas almacenado Data
#dataPath = 'C:/Users/Paula/Downloads/INSTALADORES TESIS/EmotionRecognition/Output/7525/Validation' #Cambia a la ruta donde hayas almacenado Data
dataPath = 'C:/Users/Paula/Downloads/INSTALADORES TESIS/EmotionRecognition/DatasetCopy/Dataset2.1/8020/Validation' #Cambia a la ruta donde hayas almacenado Data

# Inicialización de Vectores------------------------------------------------------------------------------
labels = []
facesData = []
predicted = []
label = 0
angry = 0
disgust = 0
fear = 0
happy = 0
sad = 0
surprise = 0
UnDetectedEmotion = 0


# Lista de Categorias (Emociones) ------------------------------------------------------------------------
emotionsList = os.listdir(dataPath)
print('Lista de emociones: ', emotionsList)

# Lectura del dataset de Validacion ----------------------------------------------------------------------
for nameDir in emotionsList:
    emotionsPath = dataPath + '/' + nameDir

    for fileName in os.listdir(emotionsPath):
        #print('Rostros: ', nameDir + '/' + fileName)
        facesData.append(cv2.imread(emotionsPath+'/'+fileName,0))
        image = cv2.imread(emotionsPath+'/'+fileName,0)
        image = imutils.resize(image, width=640)
        auxFrame = image.copy()
        faces = faceClassif.detectMultiScale(image,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(30,30),
            maxSize=(950,950))
        for (x,y,w,h) in faces:
        	labels.append(label)
        	cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)
        	rostro = auxFrame[y:y+h,x:x+w]
        	rostro = cv2.resize(rostro,(224,224),interpolation= cv2.INTER_CUBIC)
        	result = emotion_recognizer.predict(rostro)
        	predicted.append(result[0])
        	print (result)
        	print(emotionsPath+'/'+fileName)
        	if method == 'LBPH':
        		if result[1] < 70:
        			DetectedEmotion = emotionsList[result[0]]
        			if result[0] == 0: angry += 1
        			if result[0] == 1: disgust += 1
        			if result[0] == 2: fear += 1
        			if result[0] == 3: happy += 1
        			if result[0] == 4: sad += 1
        			if result[0] == 5: surprise += 1
        		else:
        			UnDetectedEmotion += 1
        	if method == 'EigenFaces':
        		if result[1] < 5700:
        			DetectedEmotion = emotionsList[result[0]]
        			if result[0] == 0: angry += 1
        			if result[0] == 1: disgust += 1
        			if result[0] == 2: fear += 1
        			if result[0] == 3: happy += 1
        			if result[0] == 4: sad += 1
        			if result[0] == 5: surprise += 1
        		else:
        			UnDetectedEmotion += 1
        	if method == 'FisherFaces':
        		if result[1] < 500:
        			DetectedEmotion = emotionsList[result[0]]
        			if result[0] == 0: angry += 1
        			if result[0] == 1: disgust += 1
        			if result[0] == 2: fear += 1
        			if result[0] == 3: happy += 1
        			if result[0] == 4: sad += 1
        			if result[0] == 5: surprise += 1
        		else:
        			UnDetectedEmotion += 1
            		
        cv2.imshow('image',image)
        cv2.waitKey(10)
    label = label + 1

# Impresion de vector con etiquetas de valor real--------------------------------------------------------
print('labels= ',labels)
print(len(labels))
print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))
print('Número de etiquetas 2: ',np.count_nonzero(np.array(labels)==2))
print('Número de etiquetas 3: ',np.count_nonzero(np.array(labels)==3))
print('Número de etiquetas 4: ',np.count_nonzero(np.array(labels)==4))
print('Número de etiquetas 5: ',np.count_nonzero(np.array(labels)==5))
print('Predicted= ',predicted)
print(len(predicted))
print('*******TOTAL EMOCIONES DETECTADAS ******** ')
print('angry=',angry)
print('disgust', disgust)
print('fear', fear)
print('happy', happy) 
print('sad', sad)
print('surprise', surprise)
print('UnDetectedEmotion', UnDetectedEmotion)

# crear matriz de confusión
matrix = confusion_matrix(labels, predicted)

# crear marco de datos de pandas Crear un conjunto de datos
dataframe = pd.DataFrame(matrix, index=emotionsList, columns=emotionsList)

print(metrics.classification_report(labels,predicted, digits = 4))

# crear mapa de calor dibujar mapa de calor
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

