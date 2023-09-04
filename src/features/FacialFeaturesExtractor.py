import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import os.path


class FacialFeaturesExtractor:

    def __init__(self, keypoints, image_path="face.jpg"):
        self.image_path = image_path
        self.keypoints = keypoints
        image = cv2.imread(image_path)
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        self.data = plt.imread(self.image_path)

    @staticmethod
    def dot(vA, vB):
        return vA[0] * vB[0] + vA[1] * vB[1]

    @staticmethod
    def ang(lineA, lineB):
        # Get nicer vector form
        vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
        vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
        # Get dot prod
        dot_prod = FacialFeaturesExtractor.dot(vA, vB)
        # Get magnitudes
        magA = FacialFeaturesExtractor.dot(vA, vA) ** 0.5
        magB = FacialFeaturesExtractor.dot(vB, vB) ** 0.5
        # Get cosine value
        cos_ = dot_prod / magA / magB
        # Get angle in radians and then convert to degrees
        angle = math.acos(dot_prod / magB / magA)
        # Basically doing angle <- angle mod 360
        # ang_deg = math.degrees(angle) % 360
        return math.degrees(angle)

    def image_size(self):
        im = cv2.imread(self.image_path)
        h, w, c = im.shape
        return w, h

    def draw_lip_line_cant(self, show=True, save_path=False):
        # left mouth corner landmark
        x1, y1 = self.keypoints[291]["X"] * self.image_width, self.keypoints[291]["Y"] * self.image_height
        # right mouth corner landmark
        x2, y2 = self.keypoints[61]["X"] * self.image_width, self.keypoints[61]["Y"] * self.image_height
        # mouth center landmark
        x3, y3 = (self.keypoints[13]["X"] + self.keypoints[14]["X"]) * self.image_width / 2, (self.keypoints[13]["Y"] + self.keypoints[14]["Y"]) * self.image_height / 2
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
        point_one = (int(self.keypoints[432]["X"] * self.image_width), int(self.keypoints[432]["Y"] * self.image_height))
        point_two = (int(self.keypoints[345]["X"] * self.image_width), int(self.keypoints[345]["Y"] * self.image_height))
        return point_one, point_two

    def get_right_cheek(self):
        point_one = (int(self.keypoints[138]["X"] * self.image_width), int(self.keypoints[138]["Y"] * self.image_height))
        point_two = (int(self.keypoints[120]["X"] * self.image_width), int(self.keypoints[120]["Y"] * self.image_height))
        return point_one, point_two

    def get_forehead(self):
        point_one = (int(self.keypoints[105]["X"] * self.image_width), int(self.keypoints[105]["Y"] * self.image_height))
        point_two = (int(self.keypoints[332]["X"] * self.image_width), int(self.keypoints[332]["Y"] * self.image_height))
        return point_one, point_two

    def get_mouth_corner_tilt(self):
        mouth_center = ((self.keypoints[13]["X"] + self.keypoints[14]["X"]) / 2, (self.keypoints[13]["Y"] + self.keypoints[14]["Y"]) / 2)
        left_mouth_tilt = 180 - FacialFeaturesExtractor.ang((mouth_center, (self.keypoints[291]["X"], self.keypoints[291]["Y"])), self.get_pupil_line())
        right_mouth_tilt = FacialFeaturesExtractor.ang((mouth_center, (self.keypoints[61]["X"], self.keypoints[61]["Y"])), self.get_pupil_line())
        if mouth_center[1] < self.keypoints[291]["Y"]:
            left_mouth_tilt = -left_mouth_tilt
        if mouth_center[1] < self.keypoints[61]["Y"]:
            right_mouth_tilt = -right_mouth_tilt
        return (round(left_mouth_tilt, 2), round(right_mouth_tilt, 2))

    def get_pupil_line(self):
        # left pupil
        x_left = [self.keypoints[385]["X"] * self.image_width, self.keypoints[387]["X"] * self.image_width,
                  self.keypoints[386]["X"] * self.image_width,
                  self.keypoints[380]["X"] * self.image_width, self.keypoints[373]["X"] * self.image_width,
                  self.keypoints[374]["X"] * self.image_width]
        y_left = [self.keypoints[385]["Y"] * self.image_height, self.keypoints[387]["Y"] * self.image_height,
                  self.keypoints[386]["Y"] * self.image_height, self.keypoints[380]["Y"] * self.image_height,
                  self.keypoints[373]["Y"] * self.image_height, self.keypoints[374]["Y"] * self.image_height]
        pupil_left = sum(x_left) / len(x_left), sum(y_left) / len(y_left)
        # right right pupil
        x_right = [self.keypoints[160]["X"] * self.image_width, self.keypoints[159]["X"] * self.image_width,
                   self.keypoints[158]["X"] * self.image_width,
                   self.keypoints[144]["X"] * self.image_width, self.keypoints[145]["X"] * self.image_width,
                   self.keypoints[153]["X"] * self.image_width]
        y_right = [self.keypoints[160]["Y"] * self.image_height, self.keypoints[159]["Y"] * self.image_height,
                   self.keypoints[158]["Y"] * self.image_height, self.keypoints[144]["Y"] * self.image_height,
                   self.keypoints[145]["Y"] * self.image_height, self.keypoints[153]["Y"] * self.image_height]
        pupil_right = sum(x_right) / len(x_right), sum(y_right) / len(y_right)
        return (pupil_left, pupil_right)

    def get_lip_ratio(self):
        return f'1:{round((self.keypoints[14]["Y"] - self.keypoints[17]["Y"]) / (self.keypoints[0]["Y"] - self.keypoints[13]["Y"]), 2)}'

    def draw_lips_ratio(self, show=True, save_path=False):
        x1, x2 = self.keypoints[0]["X"] * self.image_width, self.keypoints[13]["X"] * self.image_width
        y1, y2 = self.keypoints[0]["Y"] * self.image_height, self.keypoints[13]["Y"] * self.image_height
        x3, x4 = self.keypoints[14]["X"] * self.image_width, self.keypoints[17]["X"] * self.image_width
        y3, y4 = self.keypoints[14]["Y"] * self.image_height, self.keypoints[17]["Y"] * self.image_height

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
        x1, x2 = self.keypoints[334]["X"] * self.image_width, self.keypoints[105]["X"] * self.image_width
        y1, y2 = self.keypoints[334]["Y"] * self.image_height, self.keypoints[105]["Y"] * self.image_height
        # eyebrow medial points
        x3, x4 = self.keypoints[285]["X"] * self.image_width, self.keypoints[55]["X"] * self.image_width
        y3, y4 = self.keypoints[285]["Y"] * self.image_height, self.keypoints[55]["Y"] * self.image_height

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
        left_brow_tilt = FacialFeaturesExtractor.ang(((self.keypoints[334]["X"], self.keypoints[334]["Y"]), (self.keypoints[285]["X"], self.keypoints[285]["Y"])),
                             self.get_pupil_line())
        right_brow_tilt = 180 - FacialFeaturesExtractor.ang(
            ((self.keypoints[105]["X"], self.keypoints[105]["Y"]), (self.keypoints[55]["X"], self.keypoints[55]["Y"])),
            self.get_pupil_line())
        return (round(left_brow_tilt, 2), round(right_brow_tilt, 2))


    def draw_brow_apex_projection(self, show=True, save_path=False):
        # eyebrow apex
        x1, x2 = self.keypoints[334]["X"] * self.image_width, self.keypoints[105]["X"] * self.image_width
        y1, y2 = self.keypoints[334]["Y"] * self.image_height, self.keypoints[105]["Y"] * self.image_height
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
        # draw points
        # plt.plot(x1, y1, marker='o', color="red")
        # plt.plot(x2, y2, marker='o', color="red")
        # plt.plot(x3, y3, marker='o', color="red")
        # plt.plot(x4, y4, marker='o', color="red")
        # plt.plot(x5, y5, marker='o', color="red")
        # plt.plot(x6, y6, marker='o', color="red")
        # plt.plot(x7, y7, marker='o', color="red")
        # plt.plot(x8, y8, marker='o', color="red")
        # draw lines
        plt.plot([x1, x7], [y1, y7], color="red", linewidth=3)
        plt.plot([x2, x8], [y2, y8], color="red", linewidth=3)
        plt.plot([x3, x4], [y3, y4], color="red", linewidth=3)
        plt.plot([x5, x6], [y5, y6], color="red", linewidth=3)

        plt.axis('off')
        plt.text(x1, y1, f'{self.get_brow_apex_ratio()[0]}', fontsize=14, color="red")
        plt.text(x2, y2, f'{self.get_brow_apex_ratio()[1]}', fontsize=14, color="red")
        if not save_path:
            plt.savefig('brow_apex_projection.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'brow_apex_projection.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def get_brow_apex_projection(self):
        # left eyebrow
        p1 = np.array([self.get_pupil_line()[0][0], self.get_pupil_line()[0][1]])
        p2 = np.array([self.keypoints[263]["X"] * self.image_width, self.keypoints[263]["Y"] * self.image_height])
        p3 = np.array([self.keypoints[334]["X"] * self.image_width, self.keypoints[334]["Y"] * self.image_height])

        l2_left = np.sum((p1 - p2) ** 2)
        # if you need the point to project on line extention connecting p1 and p2
        t_left = np.sum((p3 - p1) * (p2 - p1)) / l2_left
        # if you need to ignore if p3 does not project onto line segment
        # if t_left > 1 or t_left < 0:
        #     print('p3 does not project onto p1-p2 line segment')
        projection_left = p1 + t_left * (p2 - p1)

        # right eyebrow
        p4 = np.array([self.get_pupil_line()[1][0], self.get_pupil_line()[1][1]])
        p5 = np.array([self.keypoints[33]["X"] * self.image_width, self.keypoints[33]["Y"] * self.image_height])
        p6 = np.array([self.keypoints[105]["X"] * self.image_width, self.keypoints[105]["Y"] * self.image_height])

        l2_right = np.sum((p4 - p5) ** 2)
        # if you need the point to project on line extention connecting p1 and p2
        t_right = np.sum((p6 - p4) * (p5 - p4)) / l2_right
        # if you need to ignore if p3 does not project onto line segment
        # if t_right > 1 or t_right < 0:
        #     print('p6 does not project onto p4-p5 line segment')
        projection_right = p4 + t_right * (p5 - p4)
        return [projection_left, projection_right]

    def get_brow_apex_ratio(self):
        # left eye
        p1 = np.array([self.get_pupil_line()[0][0], self.get_pupil_line()[0][1]])
        p2 = np.array([self.keypoints[263]["X"] * self.image_width, self.keypoints[263]["Y"] * self.image_height])
        p3 = self.get_brow_apex_projection()[0]
        dist_left = np.linalg.norm(p1 - p2)
        dist_to_pupil_left = np.linalg.norm(p1 - p3)
        left_ratio = dist_to_pupil_left / dist_left
        # right eye
        p4 = np.array([self.get_pupil_line()[1][0], self.get_pupil_line()[1][1]])
        p5 = np.array([self.keypoints[33]["X"] * self.image_width, self.keypoints[33]["Y"] * self.image_height])
        p6 = self.get_brow_apex_projection()[1]
        dist_right = np.linalg.norm(p4 - p5)
        dist_to_pupil_right = np.linalg.norm(p4 - p6)
        right_ratio = dist_to_pupil_right / dist_right
        return [round(left_ratio, 2), round(right_ratio, 2)]

    def draw_upper_lip_ratio(self, show=True, save_path=False):
        x1, x2 = self.keypoints[14]["X"] * self.image_width, self.keypoints[152]["X"] * self.image_width
        y1, y2 = self.keypoints[14]["Y"] * self.image_height, self.keypoints[152]["Y"] * self.image_height
        x3, x4 = self.keypoints[2]["X"] * self.image_width, self.keypoints[13]["X"] * self.image_width
        y3, y4 = self.keypoints[2]["Y"] * self.image_height, self.keypoints[13]["Y"] * self.image_height

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
        plt.text(x1 * 1.15, y1 * 1.1, f'{self.get_upper_lip_ratio()}', fontsize=22, color="red")

        if not save_path:
            plt.savefig('upper_lip_ratio.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'upper_lip_ratio.png'), bbox_inches='tight', pad_inches=0)
        if show:
            plt.show()
        plt.close()

    def get_upper_lip_ratio(self):
        return f'1:{round((self.keypoints[14]["Y"] - self.keypoints[152]["Y"]) / (self.keypoints[2]["Y"] - self.keypoints[13]["Y"]), 2)}'

    def draw_canthal_tilt(self, show=True, save_path=False):
        x1, x2 = self.keypoints[362]["X"] * self.image_width, self.keypoints[263]["X"] * self.image_width
        y1, y2 = self.keypoints[362]["Y"] * self.image_height, self.keypoints[263]["Y"] * self.image_height
        x3, x4 = self.keypoints[33]["X"] * self.image_width, self.keypoints[133]["X"] * self.image_width
        y3, y4 = self.keypoints[33]["Y"] * self.image_height, self.keypoints[133]["Y"] * self.image_height

        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)
        # draw lines
        plt.plot([x1, x2], [y1, y2], color="red", linewidth=2)
        plt.plot([x3, x4], [y3, y4], color="red", linewidth=2)

        plt.axis('off')
        plt.text(x1, y1 * 1.15, f'{self.get_canthal_tilt()[0]}°', fontsize=22, color="red")
        plt.text(x3, y3 * 1.15, f'{self.get_canthal_tilt()[1]}°', fontsize=22, color="red")
        if not save_path:
            plt.savefig('canthal_tilt.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'canthal_tilt.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def get_canthal_tilt(self):
        left_eye_tilt = FacialFeaturesExtractor.ang(((self.keypoints[362]["X"], self.keypoints[362]["Y"]), (self.keypoints[263]["X"], self.keypoints[263]["Y"])),
                            self.get_pupil_line())
        right_eye_tilt = FacialFeaturesExtractor.ang(((self.keypoints[33]["X"], self.keypoints[33]["Y"]), (self.keypoints[133]["X"], self.keypoints[133]["Y"])),
                             self.get_pupil_line())
        return (round(180 - left_eye_tilt, 2), round(180 - right_eye_tilt, 2))

    def draw_bigonial_bizygomatic_ratio(self, show=True, save_path=False):
        x1, x2 = self.keypoints[172]["X"] * self.image_width, self.keypoints[397]["X"] * self.image_width
        y1, y2 = self.keypoints[172]["Y"] * self.image_height, self.keypoints[397]["Y"] * self.image_height
        x3, x4 = self.keypoints[234]["X"] * self.image_width, self.keypoints[454]["X"] * self.image_width
        y3, y4 = self.keypoints[234]["Y"] * self.image_height, self.keypoints[454]["Y"] * self.image_height

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
        x_centers = [self.keypoints[172]["X"] * self.image_width, self.keypoints[397]["X"] * self.image_width,
                     self.keypoints[234]["X"] * self.image_width, self.keypoints[454]["X"] * self.image_width]
        y_centers = [self.keypoints[172]["Y"] * self.image_height, self.keypoints[397]["Y"] * self.image_height,
                     self.keypoints[234]["Y"] * self.image_height, self.keypoints[454]["Y"] * self.image_height]
        center_point = sum(x_centers) / len(x_centers), sum(y_centers) / len(y_centers)
        plt.text(center_point[0], center_point[1], f'{self.get_bigonial_bizygomatic_ratio()}', fontsize=22, color="red")
        if not save_path:
            plt.savefig('bigonial_bizygomatic_ratio.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'bigonial_bizygomatic_ratio.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def get_bigonial_bizygomatic_ratio(self):
        return f'{round((self.keypoints[172]["X"] - self.keypoints[397]["X"]) / (self.keypoints[234]["X"] - self.keypoints[454]["X"]), 2)}'

    def get_massage_points(self):
        return [
            (self.keypoints[357]["X"] * self.image_width, self.keypoints[357]["Y"] * self.image_height),
            (self.keypoints[350]["X"] * self.image_width, self.keypoints[350]["Y"] * self.image_height),
            (self.keypoints[348]["X"] * self.image_width, self.keypoints[348]["Y"] * self.image_height),
            (self.keypoints[347]["X"] * self.image_width, self.keypoints[347]["Y"] * self.image_height),
            (self.keypoints[346]["X"] * self.image_width, self.keypoints[346]["Y"] * self.image_height),
            (self.keypoints[264]["X"] * self.image_width, self.keypoints[264]["Y"] * self.image_height)
        ]

    def get_middle_third_massage_9_points(self):
        points_left = [357, 350, 348, 347, 346, 264, 357, 349, 330, 425, 416, 397]
        left_hand = [(self.keypoints[i]["X"] * self.image_width, self.keypoints[i]["Y"] * self.image_height) for i in points_left]

        points_right = [128, 121, 120, 119, 117, 34, 128, 120, 101, 205, 192, 172]
        right_hand = [(self.keypoints[i]["X"] * self.image_width, self.keypoints[i]["Y"] * self.image_height) for i in points_right]

        return (left_hand, right_hand)

    def get_lower_third_massage_1_points(self):
        points_left = [377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
        left_hand = [(self.keypoints[i]["X"] * self.image_width, self.keypoints[i]["Y"] * self.image_height) for i in points_left]

        points_right = [148, 176, 149, 150, 136, 172, 58, 132, 93, 234]
        right_hand = [(self.keypoints[i]["X"] * self.image_width, self.keypoints[i]["Y"] * self.image_height) for i in points_right]

        return (left_hand, right_hand)

    def draw_lower_blepharoplasy_marking(self, show=True, save_path=False):
        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)
        # линия разреза слева
        points = [252, 253, 254, 339, 255, 359, 446]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="red", linewidth=1, linestyle='dashed')

        # область коррекции слева
        points = [446, 261, 346, 347, 348, 349, 350, 357, 341, 256]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="green", linewidth=1, linestyle='dashed')

        # линия разреза справа
        points = [22, 23, 24, 110, 25, 130, 226]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="red", linewidth=1, linestyle='dashed')

        # область коррекции справа
        points = [226, 31, 117, 118, 119, 120, 121, 128, 112, 26]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="green", linewidth=1, linestyle='dashed')

        plt.axis('off')
        if not save_path:
            plt.savefig('lower_blepharoplasty.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'lower_blepharoplasty.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def draw_lower_third_threads(self, show=True, save_path=False):

        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)
        # 1-ая линия слева
        points = [400, 378, 379, 365, 397, 288, 361, 323, 454, 356]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="green", linewidth=1, linestyle='dashed')

        plt.axis('off')
        # 2-ая линия слева
        points = [369, 395, 394, 364, 367, 435, 401, 366, 447, 356]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="green", linewidth=1, linestyle='dashed')

        plt.axis('off')
        # 3-я линия слева
        points = [262, 431, 430, 434, 416, 376, 447, 356]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="green", linewidth=1, linestyle='dashed')

        plt.axis('off')
        # 4-я линия слева
        points = [424, 422, 432, 411, 352, 447, 356]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="green", linewidth=1, linestyle='dashed')

        plt.axis('off')

        if not save_path:
            plt.savefig('lower_third_threads.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'lower_third_threads.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def draw_cheek_filler_marking(self, show=True, save_path=False):

        # load image and set size in pixels
        px = 1 / plt.rcParams['figure.dpi']
        plt.figure(figsize=(self.image_width * px, self.image_height * px))
        plt.imshow(self.data)

        # область коррекции слева
        points = [345, 346, 347, 280, 352, 345]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="green", linewidth=1, linestyle='dashed')

        plt.axis('off')
        # область коррекции справа
        points = [116, 117, 118, 50, 123, 116]
        x = [self.keypoints[i]["X"] * self.image_width for i in points]
        y = [self.keypoints[i]["Y"] * self.image_height for i in points]
        plt.plot(x, y, color="green", linewidth=1, linestyle='dashed')

        plt.axis('off')
        if not save_path:
            plt.savefig('cheek_filler.png', bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, 'cheek_filler.png'), bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
