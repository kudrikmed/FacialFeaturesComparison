import os.path
import pandas as pd
import click
import logging
from pathlib import Path


@click.command()
def df_to_race_gender():
    """
    Extract and save subsets of facial features data based on race and gender categories.
    """
    # Get project root directory
    root_dir = Path(__file__).parent.parent if os.getcwd() != Path(__file__).parent.parent else os.getcwd()

    # Set up logging
    logger_file_path = os.path.join(root_dir, "logs/df_to_race_gender.log")
    logging.basicConfig(filename=logger_file_path,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        filemode="a")
    logger = logging.getLogger(__name__)
    logger.info("Feature selection starts...")

    # Load data
    data_file_path = os.path.join(root_dir, "data/processed/facial_features.csv")
    df = pd.read_csv(data_file_path)
    logger.info("Data loaded...")

    # Drop insignificant columns and target column
    df.drop("Unnamed: 0", axis=1, inplace=True)

    # Define race and gender subsets
    race_gender_subsets = {
        'Caucasian Female': 'CF',
        'Caucasian Male': 'CM',
        'Asian Female': 'AF',
        'Asian Male': 'AM'
    }

    # Process and save subsets
    for subset_name, subset_identifier in race_gender_subsets.items():
        subset_df = df[df['Filename'].str.contains(subset_identifier)]
        subset_output_path = os.path.join(root_dir, f"data/processed/{subset_identifier.lower()}_facial_features.csv")
        subset_df.to_csv(subset_output_path)
        logger.info(f"{subset_name} dataset successfully saved as {subset_output_path}")


if __name__ == "__main__":
    df_to_race_gender()
