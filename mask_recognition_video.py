from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


import pygame
import numpy as np
import imutils
import cv2
import os
from tensorflow.keras.models import load_model
import pyttsx3
import sys

class MaskDetector:
    def __init__(self, faceNet, maskNet, mask_threshold=0.9):
        self.faceNet = faceNet
        self.maskNet = maskNet
        self.mask_threshold = mask_threshold
        self.engine = pyttsx3.init()

    def say_message(self, message):
        self.engine.say(message)
        self.engine.runAndWait()

    def detect_and_predict_mask(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (244, 244), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (244, 244))
                face = np.expand_dims(face, axis=0)
                face = face / 255.0

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.vstack(faces)
            preds = self.maskNet.predict(faces, batch_size=32)

        return locs, preds

    def exit_program_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Exiting program...")
            sys.exit(0)

def load_models():
    PROTOTXT_PATH = "face_detector/deploy.prototxt"
    WEIGHTS_PATH = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    MASK_MODEL_PATH = "model_mask_detector.h5"

    prototxtPath = os.path.join(os.getcwd(), PROTOTXT_PATH)
    weightsPath = os.path.join(os.getcwd(), WEIGHTS_PATH)

    try:
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    except Exception as e:
        print(f"Error loading face detection model: {e}")
        sys.exit(1)

    try:
        maskNet = load_model(os.path.join(os.getcwd(), MASK_MODEL_PATH))
    except Exception as e:
        print(f"Error loading mask model: {e}")
        sys.exit(1)

    return faceNet, maskNet

def display_results(frame, locs, preds, people_count, mask_count, no_mask_count, mask_detector=None):
    for i, (box, pred) in enumerate(zip(locs, preds)):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask and mask > 0.9 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        if label.startswith("No Mask"):
            mask_detector.say_message("Proszę założyć maskę!")

    # Wyświetlanie licznika w rogu
    cv2.putText(frame, f"People: {people_count} Mask: {mask_count} No Mask: {no_mask_count}", (10, 30),
                cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)

def main():
    faceNet, maskNet = load_models()
    mask_detector = MaskDetector(faceNet, maskNet)

    print("[INFO] starting video stream...")

    pygame.init()
    vs = cv2.VideoCapture(0)

    # Inicjalizacja liczników
    people_count = 0
    mask_count = 0
    no_mask_count = 0


    cv2.namedWindow("Frame")

    while True:
        ret, frame = vs.read()
        frame = imutils.resize(frame, width=700)

        (locs, preds) = mask_detector.detect_and_predict_mask(frame)

        # Aktualizacja liczników
        mask_count = sum(1 for pred in preds if pred[0] > 0.9)
        no_mask_count = len(preds) - mask_count
        people_count = len(locs)

        display_results(frame, locs, preds, people_count, mask_count, no_mask_count, mask_detector)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.release()

if __name__ == "__main__":
    main()
