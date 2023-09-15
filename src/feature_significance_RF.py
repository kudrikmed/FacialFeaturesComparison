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
@click.argument('dataset', type=click.Choice(['af', 'am', 'cf', 'cm', 'all']))
def feature_selection(dataset):
    """
    Perform feature selection and evaluation for a facial features dataset.

    Args:
        dataset (str): The dataset identifier ('af', 'am', 'cf', 'cm', or 'all').
    """
    # Get project root directory
    root_dir = Path(__file__).parent.parent if os.getcwd() != Path(__file__).parent.parent else os.getcwd()

    # Define dataset and output paths based on the selected dataset
    dataset_paths = {
        'af': ("data/processed/af_facial_features.csv", "reports/figures/af_feature_significance_RF.png"),
        'am': ("data/processed/am_facial_features.csv", "reports/figures/am_feature_significance_RF.png"),
        'cf': ("data/processed/cf_facial_features.csv", "reports/figures/cf_feature_significance_RF.png"),
        'cm': ("data/processed/cm_facial_features.csv", "reports/figures/cm_feature_significance_RF.png"),
        'all': ("data/processed/facial_features.csv", "reports/figures/feature_significance_RF.png"),
    }

    dataset_path, output_plot_path = dataset_paths[dataset]

    # Set up logging
    logger_file_path = os.path.join(root_dir, "logs/evaluate_features.log")
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logger_file_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        filemode="a"
    )
    logger.info(f"Feature selection for dataset {dataset} starts...")

    # Load data
    data_file_path = os.path.join(root_dir, dataset_path)
    df = pd.read_csv(data_file_path)
    logger.info("Data loaded...")

    # Select labels
    label = df["Rating"]

    # Drop insignificant columns and target column
    drop_columns = [
        "Unnamed: 0",
        "Filename",
        "Rating",
        "Left_mouth_tilt",
        "Right_mouth_tilt",
        "Left_eyebrow_tilt",
        "Right_eyebrow_tilt",
        "Left_eyebrow_apex_ratio",
        "Right_eyebrow_apex_ratio",
        "Left_canthal_tilt",
        "Right_canthal_tilt",
    ]
    df.drop(columns=drop_columns, inplace=True)

    # Make a bar plot of label distribution
    label.value_counts().plot(kind="bar")

    # Encode labels
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)

    # Normalize features
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm = scaler.fit_transform(df_norm)
    logger.info("Data normalized...")

    # Create a Random Forest classifier
    clf = RandomForestClassifier()
    clf.fit(df_norm, label)
    logger.info("Classifier built...")

    # Evaluate the model
    logger.info("Classifier evaluation...")
    scoring_metrics = ['accuracy', 'recall_macro', 'precision_macro']
    scores = {}
    for metric in scoring_metrics:
        scores[metric] = cross_val_score(clf, df_norm, label, cv=5, scoring=metric)
    logger.info(f"Scores are: {scores}")

    # Draw a feature importance diagram
    plt.figure(figsize=(12, 12))
    plt.bar(df.columns, clf.feature_importances_)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(root_dir, output_plot_path))
    logger.info("Plot successfully saved...")


if __name__ == "__main__":
    feature_selection()
