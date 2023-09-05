import os.path
from LandmarksGenerator import LandmarksGenerator
from FacialFeaturesExtractor import FacialFeaturesExtractor
import pandas as pd
import click
import logging
from pathlib import Path


@click.command()
def build_features():
    # get project root directory
    if os.getcwd() == Path(__file__).parent.parent.parent:
        root_dir = os.getcwd()
    else:
        root_dir = Path(__file__).parent.parent.parent
    # logging
    logger = logging.getLogger(__name__)
    logger_file_path = os.path.join(root_dir, "logs/build_features.log")
    logging.basicConfig(filename=logger_file_path,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        filemode="a")
    logger.info("Feature generating starts...")
    # load rating marks from file
    rating_file_path = os.path.join(root_dir, "data/raw/All_Ratings.xlsx")
    ratings = pd.read_excel(rating_file_path)
    logger.info(f"Rating file loaded from {rating_file_path}...")
    # create empty dataframe
    output_df = pd.DataFrame(columns=['Filename',
                                      'Left_mouth_tilt',
                                      'Right_mouth_tilt',
                                      'Mean_mouth_tilt',
                                      'Lips_ratio',
                                      'Left_eyebrow_tilt',
                                      'Right_eyebrow_tilt',
                                      'Mean_eyebrow_tilt',
                                      'Left_eyebrow_apex_ratio',
                                      'Right_eyebrow_apex_ratio',
                                      'Mean_eyebrow_apex_ratio',
                                      'Upper_lip_ratio',
                                      'Left_canthal_tilt',
                                      'Right_canthal_tilt',
                                      'Mean_canthal_tilt',
                                      'Bigonial_bizygomatic_ratio',
                                      'Rating'])
    logger.info("Dataset created...")
    # set directory for images
    directory = os.path.join(root_dir, "data/raw/Images/")
    # get features for each image and add to dataframe
    image_number = 1
    for f in os.listdir(directory):
        print(f"{image_number} of {len(os.listdir(directory))} from {directory} processing...")
        filename = os.path.join(directory, f)

        # generate facial landmarks with mediapipe
        landmarks = LandmarksGenerator.landmarks_from_ready_image(filename)
        ffe = FacialFeaturesExtractor(keypoints=landmarks, image_path=filename)

        # facial features
        # mouth corners tilt
        mct = ffe.get_mouth_corner_tilt()
        left_mouth_tilt = mct[0]
        right_mouth_tilt = mct[1]
        mean_mouth_tilt = round((left_mouth_tilt + right_mouth_tilt) / 2, 2)

        # lips ratio
        lr = ffe.get_lip_ratio()
        lr = round(float(lr.split(":")[0]) / float(lr.split(":")[1]), 2)

        # medial eyebrow_tilt
        met = ffe.get_medial_eyebrow_tilt()
        left_eyebrow_tilt = met[0]
        right_eyebrow_tilt = met[1]
        mean_eyebrow_tilt = round((left_eyebrow_tilt + right_eyebrow_tilt) / 2, 2)

        # eyebrow apex ratio
        ear = ffe.get_brow_apex_ratio()
        left_eyebrow_apex_ratio = ear[0]
        right_eyebrow_apex_ratio = ear[1]
        mean_eyebrow_apex_projection = round((left_eyebrow_apex_ratio + right_eyebrow_apex_ratio) / 2, 2)

        # upper lip ratio
        ulr = ffe.get_upper_lip_ratio()
        ulr = round(float(ulr.split(":")[0]) / float(ulr.split(":")[1]), 2)

        # canthal tilt
        ct = ffe.get_canthal_tilt()
        left_canthal_tilt = ct[0]
        right_canthal_tilt = ct[1]
        mean_canthal_tilt = round((left_canthal_tilt + right_canthal_tilt) / 2, 2)

        # bigonial-bizygomatic ratio
        bbr = ffe.get_bigonial_bizygomatic_ratio()

        # get image filename
        file = os.path.basename(filename)

        # get mean rating mark
        rating = round((ratings.loc[ratings['Filename'] == file, 'Rating'].mean()), 0)

        # make a dictionary with all data
        features_dict = {'Filename': file,
                         'Left_mouth_tilt': left_mouth_tilt,
                         'Right_mouth_tilt': right_mouth_tilt,
                         'Mean_mouth_tilt': mean_mouth_tilt,
                         'Lips_ratio': lr,
                         'Left_eyebrow_tilt': left_eyebrow_tilt,
                         'Right_eyebrow_tilt': right_eyebrow_tilt,
                         'Mean_eyebrow_tilt': mean_eyebrow_tilt,
                         'Left_eyebrow_apex_ratio': left_eyebrow_apex_ratio,
                         'Right_eyebrow_apex_ratio': right_eyebrow_apex_ratio,
                         'Mean_eyebrow_apex_ratio': mean_eyebrow_apex_projection,
                         'Upper_lip_ratio': ulr,
                         'Left_canthal_tilt': left_canthal_tilt,
                         'Right_canthal_tilt': right_canthal_tilt,
                         'Mean_canthal_tilt': mean_canthal_tilt,
                         'Bigonial_bizygomatic_ratio': bbr,
                         'Rating': rating}
        # add row to output dataframe
        output_df = pd.concat([output_df, pd.DataFrame([features_dict])], ignore_index=True)
        image_number += 1

    # save as .csv
    output_path = os.path.join(root_dir, "data/processed/facial_features.csv")
    output_df.to_csv(output_path)
    logger.info(f"Full dataset successfully saved as {output_path}")


if __name__ == "__main__":
    build_features()
