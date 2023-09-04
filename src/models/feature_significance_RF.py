import os.path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import click
import logging
from src.utils.utils import get_project_root


@click.command()
def feature_selection():
    # get project root directory
    root_dir = get_project_root()
    # logging
    logger = logging.getLogger(__name__)
    logger_file_path = os.path.join(root_dir, "logs/evaluate_features.log")
    logging.basicConfig(filename=logger_file_path,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        filemode="w")
    logger.info("Feature selection starts...")
    # load data
    df = pd.read_csv(os.path.join(root_dir, "data/processed/facial_features.csv"))
    logger.info("Data loaded...")
    # select labels
    label = df["Rating"]
    # drop insignificant columns and target column
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.drop("Filename", axis=1, inplace=True)
    df.drop("Rating", axis=1, inplace=True)
    # drop interim features
    df.drop("Left_mouth_tilt", axis=1, inplace=True)
    df.drop("Right_mouth_tilt", axis=1, inplace=True)
    df.drop("Left_eyebrow_tilt", axis=1, inplace=True)
    df.drop("Right_eyebrow_tilt", axis=1, inplace=True)
    df.drop("Left_eyebrow_apex_ratio", axis=1, inplace=True)
    df.drop("Right_eyebrow_apex_ratio", axis=1, inplace=True)
    df.drop("Left_canthal_tilt", axis=1, inplace=True)
    df.drop("Right_canthal_tilt", axis=1, inplace=True)
    # make bars on plot
    label.value_counts().plot(kind="bar")
    # labels encoding
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    # normalize features
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm = scaler.fit_transform(df_norm)
    logger.info("Data normalized...")
    # create classifier
    clf = RandomForestClassifier()
    clf.fit(df_norm, label)
    logger.info("Classifier built...")
    # draw diagram
    plt.figure(figsize=(12, 12))
    plt.bar(df.columns, clf.feature_importances_)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(root_dir, "reports/figures/feature_significance_RF.png"))
    logger.info("Plot successfully saved...")


if __name__ == "__main__":
    feature_selection()
