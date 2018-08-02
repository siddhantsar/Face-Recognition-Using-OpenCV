import cv2
import numpy as np 
import pickle

if __name__ == "__main__":

	# Loading the cascade XML file using CascadeClassifier method. 
	face_cascades = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("trainner.yml")

	labels = {"person_name": 1}
	with open("labels.pickle", "rb") as f:
		og_labels= pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}

	img = cv2.imread("test-images/test-m/test-m-4.jpg")
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# For storing x, y pixels of the face in any given image
	faces = face_cascades.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)

	for x, y, w, h in faces:
		# To draw rectamgle, args: img, (starting point corner), (end point corner), (color), width)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		# Recognition (Use Deep Learning modules)
		id_, conf = recognizer.predict(roi_gray)
		print(id_)
		print(labels[id_])
		font = cv2.FONT_HERSHEY_SIMPLEX
		name = labels[id_]
		color = (255, 255, 255)
		stroke = 2
		cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		# Drawing a rectangle around a face detected
		img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

	img = cv2.resize(img, None, fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
	cv2.imshow("Output", img)
	cv2.waitKey(0)

