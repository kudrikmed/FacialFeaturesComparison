import os.path
import pandas as pd
import click
import logging
from pathlib import Path


@click.command()
def df_to_race_gender():
    # get project root directory
    if os.getcwd() == Path(__file__).parent.parent.parent:
        root_dir = os.getcwd()
    else:
        root_dir = Path(__file__).parent.parent.parent
    # logging
    logger = logging.getLogger(__name__)
    logger_file_path = os.path.join(root_dir, "logs/df_to_race_gender.log")
    logging.basicConfig(filename=logger_file_path,
                        level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        filemode="a")
    logger.info("Feature selection starts...")
    # load data
    df = pd.read_csv(os.path.join(root_dir, "data/processed/facial_features.csv"))
    logger.info("Data loaded...")
    # drop insignificant columns and target column
    df.drop("Unnamed: 0", axis=1, inplace=True)
    # dataframe subsets
    # caucasian females
    caucasian_female_df = df[df['Filename'].str.contains('CF')]
    caucasian_female_output = os.path.join(root_dir, "data/processed/cf_facial_features.csv")
    caucasian_female_df.to_csv(caucasian_female_output)
    logger.info(f"CF dataset successfully saved as {caucasian_female_output}")
    # caucasian males
    caucasian_male_df = df[df['Filename'].str.contains('CM')]
    caucasian_male_output = os.path.join(root_dir, "data/processed/cm_facial_features.csv")
    caucasian_male_df.to_csv(caucasian_male_output)
    logger.info(f"CM dataset successfully saved as {caucasian_male_output}")
    # asian females
    asian_female_df = df[df['Filename'].str.contains('AF')]
    asian_female_output = os.path.join(root_dir, "data/processed/af_facial_features.csv")
    asian_female_df.to_csv(asian_female_output)
    logger.info(f"AF dataset successfully saved as {asian_female_output}")
    # asian males
    asian_male_df = df[df['Filename'].str.contains('AM')]
    asian_male_output = os.path.join(root_dir, "data/processed/am_facial_features.csv")
    asian_male_df.to_csv(asian_male_output)
    logger.info(f"AM dataset successfully saved as {asian_male_output}")


if __name__ == "__main__":
    df_to_race_gender()
