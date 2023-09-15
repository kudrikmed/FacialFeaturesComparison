import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import os.path


class FacialFeaturesExtractor:
    """
    A class for extracting and analyzing facial features from an image.

    Attributes:
        keypoints (dict): A dictionary containing facial keypoints.
        image_path (str): The path to the input image.
    """

    def __init__(self, keypoints, image_path="face.jpg"):
        """
        Initializes the FacialFeaturesExtractor.

        Args:
            keypoints (dict): A dictionary containing facial keypoints.
            image_path (str, optional): The path to the input image. Defaults to "face.jpg".
        """
        self.image_path = image_path
        self.keypoints = keypoints
        image = cv2.imread(image_path)
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        self.data = plt.imread(self.image_path)

    @staticmethod
    def dot(vA, vB):
        """
        Calculate the dot product of two vectors.

        Args:
            vA (tuple): The first vector.
            vB (tuple): The second vector.

        Returns:
            float: The dot product of the two vectors.
        """
        return vA[0] * vB[0] + vA[1] * vB[1]

    @staticmethod
    def ang(lineA, lineB):
        """
        Calculate the angle between two lines.

        Args:
            lineA (tuple): A tuple containing two points defining the first line.
            lineB (tuple): A tuple containing two points defining the second line.

        Returns:
            float: The angle between the two lines in degrees.
        """
        vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
        vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
        dot_prod = FacialFeaturesExtractor.dot(vA, vB)
        magA = FacialFeaturesExtractor.dot(vA, vA) ** 0.5
        magB = FacialFeaturesExtractor.dot(vB, vB) ** 0.5
        cos_ = dot_prod / magA / magB
        angle = math.acos(dot_prod / magB / magA)
        return math.degrees(angle)

    def draw_lip_line_cant(self, show=True, save_path=False):
        # left mouth corner landmark
        x1 = self.keypoints[291]["X"] * self.image_width
        y1 = self.keypoints[291]["Y"] * self.image_height
        # right mouth corner landmark
        x2 = self.keypoints[61]["X"] * self.image_width
        y2 = self.keypoints[61]["Y"] * self.image_height
        # mouth center landmark
        x3 = (self.keypoints[13]["X"] + self.keypoints[14]["X"]) * self.image_width / 2
        y3 = (self.keypoints[13]["Y"] + self.keypoints[14]["Y"]) * self.image_height / 2
        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)
        # draw points
        plt.plot(x1, y1, marker='o', color="red")
        plt.plot(x2, y2, marker='o', color="red")
        plt.plot(x3, y3, marker='o', color="red")
        # draw lines
        plt.plot([x3, x1], [y3, y1], color="red", linewidth=3)
        plt.plot([x3, x2], [y3, y2], color="red", linewidth=3)

        plt.axis('off')
        plt.text(x1, y1 * 1.15, f'{self.get_mouth_corner_tilt()[0]}°', fontsize=22, color="red")
        plt.text(x2, y2 * 1.15, f'{self.get_mouth_corner_tilt()[1]}°', fontsize=22, color="red")
        if not save_path:
            plt.savefig('lips_cant.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'lips_cant.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def get_left_cheek(self):
        point_one = (int(self.keypoints[432]["X"] * self.image_width),
                     int(self.keypoints[432]["Y"] * self.image_height))
        point_two = (int(self.keypoints[345]["X"] * self.image_width),
                     int(self.keypoints[345]["Y"] * self.image_height))
        return point_one, point_two

    def get_right_cheek(self):
        point_one = (int(self.keypoints[138]["X"] * self.image_width),
                     int(self.keypoints[138]["Y"] * self.image_height))
        point_two = (int(self.keypoints[120]["X"] * self.image_width),
                     int(self.keypoints[120]["Y"] * self.image_height))
        return point_one, point_two

    def get_forehead(self):
        point_one = (int(self.keypoints[105]["X"] * self.image_width),
                     int(self.keypoints[105]["Y"] * self.image_height))
        point_two = (int(self.keypoints[332]["X"] * self.image_width),
                     int(self.keypoints[332]["Y"] * self.image_height))
        return point_one, point_two

    def get_mouth_corner_tilt(self):
        mouth_center = ((self.keypoints[13]["X"] + self.keypoints[14]["X"]) / 2,
                        (self.keypoints[13]["Y"] + self.keypoints[14]["Y"]) / 2)
        left_mouth_corner_point = (self.keypoints[291]["X"], self.keypoints[291]["Y"])
        right_mouth_corner_point = (self.keypoints[61]["X"], self.keypoints[61]["Y"])
        left_mouth_tilt = 180 - FacialFeaturesExtractor.ang((mouth_center,left_mouth_corner_point),
                                                            self.get_pupil_line())
        right_mouth_tilt = FacialFeaturesExtractor.ang((mouth_center, right_mouth_corner_point),
                                                       self.get_pupil_line())
        if mouth_center[1] < self.keypoints[291]["Y"]:
            left_mouth_tilt = -left_mouth_tilt
        if mouth_center[1] < self.keypoints[61]["Y"]:
            right_mouth_tilt = -right_mouth_tilt
        return round(left_mouth_tilt, 2), round(right_mouth_tilt, 2)

    def get_pupil_line(self):
        # left pupil
        x_left = [self.keypoints[385]["X"] * self.image_width,
                  self.keypoints[387]["X"] * self.image_width,
                  self.keypoints[386]["X"] * self.image_width,
                  self.keypoints[380]["X"] * self.image_width,
                  self.keypoints[373]["X"] * self.image_width,
                  self.keypoints[374]["X"] * self.image_width]
        y_left = [self.keypoints[385]["Y"] * self.image_height,
                  self.keypoints[387]["Y"] * self.image_height,
                  self.keypoints[386]["Y"] * self.image_height,
                  self.keypoints[380]["Y"] * self.image_height,
                  self.keypoints[373]["Y"] * self.image_height,
                  self.keypoints[374]["Y"] * self.image_height]
        pupil_left = sum(x_left) / len(x_left), sum(y_left) / len(y_left)

        # right right pupil
        x_right = [self.keypoints[160]["X"] * self.image_width,
                   self.keypoints[159]["X"] * self.image_width,
                   self.keypoints[158]["X"] * self.image_width,
                   self.keypoints[144]["X"] * self.image_width,
                   self.keypoints[145]["X"] * self.image_width,
                   self.keypoints[153]["X"] * self.image_width]
        y_right = [self.keypoints[160]["Y"] * self.image_height,
                   self.keypoints[159]["Y"] * self.image_height,
                   self.keypoints[158]["Y"] * self.image_height,
                   self.keypoints[144]["Y"] * self.image_height,
                   self.keypoints[145]["Y"] * self.image_height,
                   self.keypoints[153]["Y"] * self.image_height]
        pupil_right = sum(x_right) / len(x_right), sum(y_right) / len(y_right)
        return pupil_left, pupil_right

    def get_lip_ratio(self):
        upper = self.keypoints[14]["Y"] - self.keypoints[17]["Y"]
        lower = self.keypoints[0]["Y"] - self.keypoints[13]["Y"]
        return f'1:{round(upper / lower, 2)}'

    def draw_lips_ratio(self, show=True, save_path=False):
        x1 = self.keypoints[0]["X"] * self.image_width
        x2 = self.keypoints[13]["X"] * self.image_width
        y1 = self.keypoints[0]["Y"] * self.image_height
        y2 = self.keypoints[13]["Y"] * self.image_height
        x3 = self.keypoints[14]["X"] * self.image_width
        x4 = self.keypoints[17]["X"] * self.image_width
        y3 = self.keypoints[14]["Y"] * self.image_height
        y4 = self.keypoints[17]["Y"] * self.image_height

        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)
        # draw points
        plt.plot(x1, y1, marker='o', color="red")
        plt.plot(x2, y2, marker='o', color="red")
        plt.plot(x3, y3, marker='o', color="red")
        plt.plot(x4, y4, marker='o', color="red")
        # draw lines
        plt.plot([x1, x2], [y1, y2], color="red", linewidth=3)
        plt.plot([x3, x4], [y3, y4], color="red", linewidth=3)

        plt.axis('off')
        plt.text(x1 * 1.15, y1 * 1.1, f'{self.get_lip_ratio()}', fontsize=22, color="red")
        if not save_path:
            plt.savefig('lips_ratio.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'lips_ratio.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def draw_medial_eyebrow_tilt(self, show=True, save_path=False):
        # eyebrow apex
        x1 = self.keypoints[334]["X"] * self.image_width
        x2 = self.keypoints[105]["X"] * self.image_width
        y1 = self.keypoints[334]["Y"] * self.image_height
        y2 = self.keypoints[105]["Y"] * self.image_height
        # eyebrow medial points
        x3 = self.keypoints[285]["X"] * self.image_width
        x4 = self.keypoints[55]["X"] * self.image_width
        y3 = self.keypoints[285]["Y"] * self.image_height
        y4 = self.keypoints[55]["Y"] * self.image_height

        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)
        # draw points
        plt.plot(x1, y1, marker='o', color="red")
        plt.plot(x2, y2, marker='o', color="red")
        plt.plot(x3, y3, marker='o', color="red")
        plt.plot(x4, y4, marker='o', color="red")
        # draw lines
        plt.plot([x1, x3], [y1, y3], color="red", linewidth=3)
        plt.plot([x2, x4], [y2, y4], color="red", linewidth=3)

        plt.axis('off')
        plt.text(x1, y1, f'{self.get_medial_eyebrow_tilt()[0]}°', fontsize=22, color="red")
        plt.text(x2, y2, f'{self.get_medial_eyebrow_tilt()[1]}°', fontsize=22, color="red")
        if not save_path:
            plt.savefig('medial_eyebrow_tilt.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'medial_eyebrow_tilt.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def get_medial_eyebrow_tilt(self):
        left_brow_tilt = FacialFeaturesExtractor.ang(((self.keypoints[334]["X"], self.keypoints[334]["Y"]),
                                                     (self.keypoints[285]["X"], self.keypoints[285]["Y"])),
                                                     self.get_pupil_line())
        right_brow_tilt = 180 - FacialFeaturesExtractor.ang(((self.keypoints[105]["X"], self.keypoints[105]["Y"]),
                                                            (self.keypoints[55]["X"], self.keypoints[55]["Y"])),
                                                            self.get_pupil_line())
        return round(left_brow_tilt, 2), round(right_brow_tilt, 2)

    def draw_brow_apex_projection(self, show=True, save_path=False):
        # eyebrow apex
        x1 = self.keypoints[334]["X"] * self.image_width
        x2 = self.keypoints[105]["X"] * self.image_width
        y1 = self.keypoints[334]["Y"] * self.image_height
        y2 = self.keypoints[105]["Y"] * self.image_height
        # pupil-latheral cantus line left
        x3, y3 = self.get_pupil_line()[0][0], self.get_pupil_line()[0][1]
        x4, y4 = self.keypoints[263]["X"] * self.image_width, self.keypoints[263]["Y"] * self.image_height
        # pupil-latheral cantus line right
        x5, y5 = self.get_pupil_line()[1][0], self.get_pupil_line()[1][1]
        x6, y6 = self.keypoints[33]["X"] * self.image_width, self.keypoints[33]["Y"] * self.image_height
        # projection of left apex
        x7, y7 = self.get_brow_apex_projection()[0][0], self.get_brow_apex_projection()[0][1]
        # projection of right apex
        x8, y8 = self.get_brow_apex_projection()[1][0], self.get_brow_apex_projection()[1][1]

        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)

        # draw lines
        plt.plot([x1, x7], [y1, y7], color="red", linewidth=3)
        plt.plot([x2, x8], [y2, y8], color="red", linewidth=3)
        plt.plot([x3, x4], [y3, y4], color="red", linewidth=3)
        plt.plot([x5, x6], [y5, y6], color="red", linewidth=3)

        plt.axis('off')
        plt.text(x1,
                 y1,
                 f'{self.get_brow_apex_ratio()[0]}',
                 fontsize=22,
                 color="red")
        plt.text(x2,
                 y2,
                 f'{self.get_brow_apex_ratio()[1]}',
                 fontsize=22,
                 color="red")
        if not save_path:
            plt.savefig('brow_apex_projection.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'brow_apex_projection.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def get_brow_apex_projection(self):
        # left eyebrow
        p1 = np.array([self.get_pupil_line()[0][0],
                       self.get_pupil_line()[0][1]])
        p2 = np.array([self.keypoints[263]["X"] * self.image_width,
                       self.keypoints[263]["Y"] * self.image_height])
        p3 = np.array([self.keypoints[334]["X"] * self.image_width,
                       self.keypoints[334]["Y"] * self.image_height])

        l2_left = np.sum((p1 - p2) ** 2)
        t_left = np.sum((p3 - p1) * (p2 - p1)) / l2_left
        projection_left = p1 + t_left * (p2 - p1)

        # right eyebrow
        p4 = np.array([self.get_pupil_line()[1][0],
                       self.get_pupil_line()[1][1]])
        p5 = np.array([self.keypoints[33]["X"] * self.image_width,
                       self.keypoints[33]["Y"] * self.image_height])
        p6 = np.array([self.keypoints[105]["X"] * self.image_width,
                       self.keypoints[105]["Y"] * self.image_height])

        l2_right = np.sum((p4 - p5) ** 2)
        t_right = np.sum((p6 - p4) * (p5 - p4)) / l2_right
        projection_right = p4 + t_right * (p5 - p4)

        return [projection_left, projection_right]

    def get_brow_apex_ratio(self):
        # left eye
        p1 = np.array([self.get_pupil_line()[0][0],
                       self.get_pupil_line()[0][1]])
        p2 = np.array([self.keypoints[263]["X"] * self.image_width,
                       self.keypoints[263]["Y"] * self.image_height])
        p3 = self.get_brow_apex_projection()[0]
        dist_left = np.linalg.norm(p1 - p2)
        dist_to_pupil_left = np.linalg.norm(p1 - p3)
        left_ratio = dist_to_pupil_left / dist_left
        # right eye
        p4 = np.array([self.get_pupil_line()[1][0],
                       self.get_pupil_line()[1][1]])
        p5 = np.array([self.keypoints[33]["X"] * self.image_width,
                       self.keypoints[33]["Y"] * self.image_height])
        p6 = self.get_brow_apex_projection()[1]
        dist_right = np.linalg.norm(p4 - p5)
        dist_to_pupil_right = np.linalg.norm(p4 - p6)
        right_ratio = dist_to_pupil_right / dist_right
        return [round(left_ratio, 2), round(right_ratio, 2)]

    def draw_upper_lip_ratio(self, show=True, save_path=False):
        x1 = self.keypoints[14]["X"] * self.image_width
        x2 = self.keypoints[152]["X"] * self.image_width
        y1 = self.keypoints[14]["Y"] * self.image_height
        y2 = self.keypoints[152]["Y"] * self.image_height
        x3 = self.keypoints[2]["X"] * self.image_width
        x4 = self.keypoints[13]["X"] * self.image_width
        y3 = self.keypoints[2]["Y"] * self.image_height
        y4 = self.keypoints[13]["Y"] * self.image_height

        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)
        # draw points
        plt.plot(x1, y1, marker='o', color="red")
        plt.plot(x2, y2, marker='o', color="red")
        plt.plot(x3, y3, marker='o', color="red")
        plt.plot(x4, y4, marker='o', color="red")
        # draw lines
        plt.plot([x1, x2], [y1, y2], color="red", linewidth=3)
        plt.plot([x3, x4], [y3, y4], color="red", linewidth=3)

        plt.axis('off')
        plt.text(x1 * 1.15,
                 y1 * 1.1,
                 f'{self.get_upper_lip_ratio()}',
                 fontsize=22,
                 color="red")

        if not save_path:
            plt.savefig('upper_lip_ratio.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'upper_lip_ratio.png'), bbox_inches='tight', pad_inches=0)
        if show:
            plt.show()
        plt.close()

    def get_upper_lip_ratio(self):
        upper = self.keypoints[14]["Y"] - self.keypoints[152]["Y"]
        lower = self.keypoints[2]["Y"] - self.keypoints[13]["Y"]
        return f'1:{round(upper / lower, 2)}'

    def draw_canthal_tilt(self, show=True, save_path=False):
        x1 = self.keypoints[362]["X"] * self.image_width
        x2 = self.keypoints[263]["X"] * self.image_width
        y1 = self.keypoints[362]["Y"] * self.image_height
        y2 = self.keypoints[263]["Y"] * self.image_height
        x3 = self.keypoints[33]["X"] * self.image_width
        x4 = self.keypoints[133]["X"] * self.image_width
        y3 = self.keypoints[33]["Y"] * self.image_height
        y4 = self.keypoints[133]["Y"] * self.image_height

        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)
        # draw lines
        plt.plot([x1, x2], [y1, y2], color="red", linewidth=2)
        plt.plot([x3, x4], [y3, y4], color="red", linewidth=2)

        plt.axis('off')
        plt.text(x1,
                 y1 * 1.15,
                 f'{self.get_canthal_tilt()[0]}°',
                 fontsize=22,
                 color="red")
        plt.text(x3,
                 y3 * 1.15,
                 f'{self.get_canthal_tilt()[1]}°',
                 fontsize=22,
                 color="red")
        if not save_path:
            plt.savefig('canthal_tilt.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'canthal_tilt.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def get_canthal_tilt(self):
        left_eye_tilt = FacialFeaturesExtractor.ang(((self.keypoints[362]["X"], self.keypoints[362]["Y"]),
                                                     (self.keypoints[263]["X"], self.keypoints[263]["Y"])),
                                                    self.get_pupil_line())
        right_eye_tilt = FacialFeaturesExtractor.ang(((self.keypoints[33]["X"], self.keypoints[33]["Y"]),
                                                      (self.keypoints[133]["X"], self.keypoints[133]["Y"])),
                                                     self.get_pupil_line())
        return round(180 - left_eye_tilt, 2), round(180 - right_eye_tilt, 2)

    def draw_bigonial_bizygomatic_ratio(self, show=True, save_path=False):
        x1 = self.keypoints[172]["X"] * self.image_width
        x2 = self.keypoints[397]["X"] * self.image_width
        y1 = self.keypoints[172]["Y"] * self.image_height
        y2 = self.keypoints[397]["Y"] * self.image_height
        x3 = self.keypoints[234]["X"] * self.image_width
        x4 = self.keypoints[454]["X"] * self.image_width
        y3 = self.keypoints[234]["Y"] * self.image_height
        y4 = self.keypoints[454]["Y"] * self.image_height

        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)
        # draw points
        plt.plot(x1, y1, marker='o', color="red")
        plt.plot(x2, y2, marker='o', color="red")
        plt.plot(x3, y3, marker='o', color="red")
        plt.plot(x4, y4, marker='o', color="red")
        # draw lines
        plt.plot([x1, x2], [y1, y2], color="red", linewidth=3)
        plt.plot([x3, x4], [y3, y4], color="red", linewidth=3)

        plt.axis('off')
        x_centers = [self.keypoints[172]["X"] * self.image_width,
                     self.keypoints[397]["X"] * self.image_width,
                     self.keypoints[234]["X"] * self.image_width,
                     self.keypoints[454]["X"] * self.image_width]
        y_centers = [self.keypoints[172]["Y"] * self.image_height,
                     self.keypoints[397]["Y"] * self.image_height,
                     self.keypoints[234]["Y"] * self.image_height,
                     self.keypoints[454]["Y"] * self.image_height]
        center_point = sum(x_centers) / len(x_centers), sum(y_centers) / len(y_centers)
        plt.text(center_point[0],
                 center_point[1],
                 f'{self.get_bigonial_bizygomatic_ratio()}',
                 fontsize=22,
                 color="red")
        if not save_path:
            plt.savefig('bigonial_bizygomatic_ratio.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'bigonial_bizygomatic_ratio.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def get_bigonial_bizygomatic_ratio(self):
        bigonial = self.keypoints[172]["X"] - self.keypoints[397]["X"]
        bizygomatic = self.keypoints[234]["X"] - self.keypoints[454]["X"]
        return f'{round(bigonial / bizygomatic, 2)}'
