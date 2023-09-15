import cv2
import mediapipe as mp


class LandmarksGenerator:
    """
    A class to generate facial landmarks from an image using the MediaPipe library.
    """

    @staticmethod
    def landmarks_from_ready_image(image_path):
        """
        Generate facial landmarks from an image using MediaPipe FaceMesh.

        Args:
            image_path (str): The path to the input image.

        Returns:
            list: A list of dictionaries containing X, Y, and Z coordinates of facial landmarks.

        Example:
            landmarks = LandmarksGenerator.landmarks_from_ready_image('image.jpg')
        """
        image_files = [image_path]
        mp_face_mesh = mp.solutions.face_mesh

        # Initialize the FaceMesh model
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5) as face_mesh:
            for idx, file in enumerate(image_files):
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                landmarks = []

                # Extract face mesh landmarks if detected
                if not results.multi_face_landmarks:
                    continue

                for data_point in results.multi_face_landmarks[0].landmark:
                    landmarks.append({
                        'X': data_point.x,
                        'Y': data_point.y,
                        'Z': data_point.z
                    })

        return landmarks
