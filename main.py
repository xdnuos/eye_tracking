import cv2
import numpy as np
import tensorflow as tf
import os
from postprocessing import parse_heatmaps
import dlib
from imutils import face_utils

devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
model = tf.keras.models.load_model("./exported/hrnet-gaze")

dirname = os.path.dirname(__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(dirname, 'lbpcascade_frontalface_improved.xml'))
landmarks_detector = dlib.shape_predictor(os.path.join(dirname, 'shape_predictor_5_face_landmarks.dat'))

def normalize(inputs):
    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Normalization
    return ((inputs / 255.0) - img_mean)/img_std
def detect_landmarks(face, frame, scale_x=0, scale_y=0):
    (x, y, w, h) = (int(e) for e in face)
    rectangle = dlib.rectangle(x, y, x + w, y + h)
    face_landmarks = landmarks_detector(frame, rectangle)
    return face_utils.shape_to_np(face_landmarks)

def draw_landmarks(frame, landmarks):
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
def segment_eyes(frame, landmarks, w=256, h=256):
    eye = []
            # [[423.94749342 321.00236844] 0 R
            # [395.94999313 324.00249344] 1 R
            # [326.89999342 326.99999375] 2 L
            # [355.89986844 327.00248749] 3 L
            # [378.89749313 375.05248719]] 4 C
    x1, y1 = landmarks[0]
    x2, y2 = landmarks[1]
    eye_width = (x1-x2)+(x1-x2)/2
    if eye_width == 0.0:
        return eye
    eye = frame[int(y2-eye_width/1.7):int(y2+eye_width-eye_width/1.7), int(x2-eye_width/8):int(x2 + eye_width-eye_width/8)]
    eye = cv2.resize(eye,[256,256])
    return eye
def main():
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    webcam.set(cv2.CAP_PROP_FPS, 30)
    current_face = None
    landmarks = None
    alpha = 0.95
    while True:
        frame_got, frame = webcam.read()
        if frame_got is False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        if len(faces):
            next_face = faces[0]
            if current_face is not None:
                current_face = alpha * next_face + (1 - alpha) * current_face
            else:
                current_face = next_face

        if current_face is not None:
            next_landmarks = detect_landmarks(current_face, gray)

            if landmarks is not None:
                landmarks = next_landmarks * alpha + (1 - alpha) * landmarks
            else:
                landmarks = next_landmarks
            # draw_landmarks(landmarks, frame)
        if landmarks is not None:
            eye = segment_eyes(frame, landmarks)
            copy_eye = eye.copy()
            # height, width, channels = eye.shape
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
            eye = normalize(np.array(eye, dtype=np.float32))
            eye = np.expand_dims(eye,axis=0)
            eye = np.array(eye, dtype=np.float32)

            heatmap_group = model.predict(eye)
            marks, _ = parse_heatmaps(heatmap_group[0], (256, 256))
            draw_landmarks(copy_eye, [marks[20]])
            cv2.imshow("Eye", copy_eye)
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(10)
        if key == 27:
            break
    webcam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()

