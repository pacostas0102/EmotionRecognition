import cv2
import os
import imutils
import time

inicio= time.time()

#Emotions to Capture ----------------------------------------------------------------------------------------------------------
emotionName = 'angry'
#emotionName = 'disgust'
#emotionName = 'fear'
#emotionName = 'happy'
#emotionName = 'sad'
#emotionName = 'surprise'

#PreTrain Model----------------------------------------------------------------------------------------------------------------
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


#Data Path To Read----------------------------------------------------------------------------------------------------------------
#image = cv2.imread('C:/Users/Paula/Downloads/INSTALADORES TESIS/Reconocimiento Emociones/DATA/DATASETFULL/pruebahappy.jpg')
#image = cv2.imread('C:/Users/Paula/Downloads/INSTALADORES TESIS/Reconocimiento Emociones/DATA/DATASETFULL/train/happy/95.jpeg')
#dataPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/EmotionRecognition/DatasetCopy/Dataset2.1/7030"
#dataPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/EmotionRecognition/DatasetCopy/Dataset2.1/7525"
#dataPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/EmotionRecognition/DatasetCopy/Dataset2.1/8020"
#dataPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/FaceRecognition/Dataset2.0/7030/Train"
#dataPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/FaceRecognition/Dataset2.0/7525/Train"
#dataPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/FaceRecognition/Dataset2.0/8020/Train"
dataPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/FaceRecognition/Dataset2.0/7030/Validation"
#dataPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/FaceRecognition/Dataset2.0/7525/Validation"
#dataPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/FaceRecognition/Dataset2.0/7525/Validation"

input_images_path = dataPath + '/' + emotionName

files_names = os.listdir(input_images_path)
print(len(files_names))
print(files_names)

#Data Path To Write----------------------------------------------------------------------------------------------------------------
#dataOPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/Output/7030/Train"
#dataOPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/Output/7525/Train"
#dataOPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/Output/8020/Train"
#dataOPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/Output/7030/Validation"
#dataOPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/Output/7525/Validation"
#dataOPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/Output/8020/Validation"
dataOPath = "C:/Users/Paula/Downloads/INSTALADORES TESIS/EstuHaarCascade/100_1000"

output_images_path = dataOPath + '/' + emotionName

if not os.path.exists(output_images_path):
    os.makedirs(output_images_path)
    print("Directorio creado: ", output_images_path)
count = 0
fotos = 0


for file_name in files_names:
    image_path = input_images_path + "/" + file_name
    print(image_path)
    image = cv2.imread(image_path)
    if image is None:continue
    image =	imutils.resize(image, width=640)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    auxFrame = image.copy()
    faces = faceClassif.detectMultiScale(gray,
  		scaleFactor=1.1,
  		minNeighbors=5,
  		minSize=(100,100),
  		maxSize=(1000,1000))
    for (x,y,w,h) in faces:
    	cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)
    	rostro = auxFrame[y:y+h,x:x+w]
    	rostro = cv2.resize(rostro,(224,224),interpolation=cv2.INTER_CUBIC)
    	cv2.imwrite(output_images_path + '/{}.jpg'.format(count),rostro)
    	count = count + 1
    fotos +=1
    #cv2.imshow('frame',image)
fin=time.time()
print('fotos',fotos)
print(fin-inicio)
#Mostrar Imagen
#cv2.imshow('imagen',image)
cv2.waitKey(20)
"""
#Redimencionamiento proporcional
(high, width, channels) = image.shape
print('Alto={}, Ancho={}, Canales={}'.format(high, width, channels))
r= 200/width
dim = (200,int(high*r))
redimp = cv2.resize(image,dim)
#Mostrar Imagen
cv2.imshow('imagen',redimp)
cv2.waitKey(0)


image =	imutils.resize(image, width=640)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
auxFrame = image.copy()
faces = faceClassif.detectMultiScale(gray,
  scaleFactor=1.1,
  minNeighbors=5,
  minSize=(30,30),
  maxSize=(800,800))

for (x,y,w,h) in faces:
	#cv2.rectangle(image, start_point, end_point, color, thickness)
	cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)


#Mostrar Imagen
cv2.imshow('imagen',image)
cv2.waitKey(0)
"""

"""
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(gray,
  scaleFactor=1.1,
  minNeighbors=5,
  minSize=(30,30),
  maxSize=(200,200))

for (x,y,w,h) in faces:
  cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
