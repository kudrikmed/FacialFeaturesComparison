import os.path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import click
import logging
from pathlib import Path


@click.command()
@click.argument('dataset', type=click.types.STRING)
def feature_selection(dataset):
    # get project root directory
    if os.getcwd() == Path(__file__).parent.parent.parent:
        root_dir = os.getcwd()
    else:
        root_dir = Path(__file__).parent.parent.parent
    # select dataset
    if dataset == 'af':
        dataset_path = "data/processed/af_facial_features.csv"
        output_plot_path = "reports/figures/af_feature_significance_RF.png"
    elif dataset == 'am':
        dataset_path = 'data/processed/am_facial_features.csv'
        output_plot_path = "reports/figures/am_feature_significance_RF.png"
    elif dataset == 'cf':
        dataset_path = 'data/processed/cf_facial_features.csv'
        output_plot_path = "reports/figures/cf_feature_significance_RF.png"
    elif dataset == 'cm':
        dataset_path = 'data/processed/cm_facial_features.csv'
        output_plot_path = "reports/figures/cm_feature_significance_RF.png"
    else:
        dataset_path = 'data/processed/facial_features.csv'
        output_plot_path = "reports/figures/feature_significance_RF.png"
    # logging
    logger = logging.getLogger(__name__)
    logger_file_path = os.path.join(root_dir, "logs/evaluate_features.log")
    logging.basicConfig(filename=logger_file_path,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        filemode="a")
    logger.info(f"Feature selection for dataset {dataset} starts...")
    # load data
    df = pd.read_csv(os.path.join(root_dir, dataset_path))
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
    # evaluate model
    logger.info("Classifier evaluation...")
    accuracy = cross_val_score(clf, df_norm, label, cv=5, scoring='accuracy')
    recall = cross_val_score(clf, df_norm, label, cv=5, scoring='recall_macro')
    precision = cross_val_score(clf, df_norm, label, cv=5, scoring='precision_macro')
    logger.info(f"Scores are: accuracy - {accuracy}, recall_macro - {recall}, precision_macro - {precision}")
    # draw diagram
    plt.figure(figsize=(12, 12))
    plt.bar(df.columns, clf.feature_importances_)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(root_dir, output_plot_path))
    logger.info("Plot successfully saved...")


if __name__ == "__main__":
    feature_selection()
