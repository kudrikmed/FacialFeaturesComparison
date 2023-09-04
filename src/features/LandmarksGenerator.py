import matplotlib.pyplot as plt
import cv2
import mediapipe as mp


class LandmarksGenerator:

    @staticmethod
    def landmarks_from_ready_image(image_path):
        image_files = [image_path]
        plt.switch_backend('agg')
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5) as face_mesh:
            for idx, file in enumerate(image_files):
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue
                landmarks = []
                for data_point in results.multi_face_landmarks[0].landmark:
                    landmarks.append({
                        'X': data_point.x,
                        'Y': data_point.y,
                        'Z': data_point.z
                    })
        return landmarks