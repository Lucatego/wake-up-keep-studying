import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


if __name__ == "__main__":

    base_options = python.BaseOptions(model_asset_path = "./models/face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options = base_options, num_faces = 1)

    camera = cv2.VideoCapture(0)
    detector = vision.FaceLandmarker.create_from_options(options)

    print("Press ESC to exit")

    while True:
        ret, frame = camera.read()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        if result.multi_face_landmarks:
            for face in result.multi_face_landmarks:
                # Verificacion y lógica
                pass

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == 27:
            break