import cv2
import numpy as np
import pickle

# Loading the cascade xml file
face_cascades = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)
scaling_factor = 0.5
print("Press Esc to exit.")

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()

        # Resizing the frame
        frame = cv2.resize(frame, None, fx = scaling_factor, fy = scaling_factor, interpolation = cv2.INTER_AREA)

        # Converting to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecting faces within frame and getting the pixel values
        faces = face_cascades.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

        for (x, y, w, h) in faces: #print(x, y, w, h)
            roi_gray = gray[y: y + h, x: x + w]
            roi_color = frame[y: y + h, x: x + w]

            # Recognition
            id_, conf = recognizer.predict(roi_gray)

            if conf >= 45:
                print(id_)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

            img_item = "my-image.png"
            cv2.imwrite(img_item, roi_gray)

            # Drawing a rectangle around a face detected
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("Face Recognition", frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()