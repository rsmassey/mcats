
import pandas as pd
import os
from colorama import Fore, Style
from mcats.ml_logic.params import LOCAL_DATA_PATH
from sklearn.model_selection import train_test_split


def get_pandas_chunk(path: str,
                     index: int,
                     chunk_size: int,
                     dtypes,
                     columns: list = None,
                     verbose=True) -> pd.DataFrame:
    """
    return a chunk of the raw dataset from local disk
    """
    path = os.path.join(
        os.path.expanduser(LOCAL_DATA_PATH),
        "processed" if "processed" in path else "raw",
        f"{path}.csv")

    if verbose:
        print(Fore.MAGENTA + f"Source data from {path}: {chunk_size if chunk_size is not None else 'all'} rows (from row {index})" + Style.RESET_ALL)

    try:

        df = pd.read_csv(
                path,
                skiprows=index + 1,  # skip header
                nrows=chunk_size,
                dtype=dtypes,
                header=None)  # read all rows

        # read_csv(dtypes=...) will silently fail to convert data types, if column names do no match dictionnary key provided.
        if isinstance(dtypes, dict):
            assert dict(df.dtypes) == dtypes

        if columns is not None:
            df.columns = columns

    except pd.errors.EmptyDataError:

        return None  # end of data

    return df


def save_local_chunk(path: str,
                     data: pd.DataFrame,
                     is_first: bool):
    """
    save a chunk of the dataset to local disk
    """

    path = os.path.join(
        os.path.expanduser(LOCAL_DATA_PATH),
        "raw" if "raw" in path else "processed",
        f"{path}.csv")

    print(Fore.BLUE + f"\nSave data to {path}:" + Style.RESET_ALL)

    data.to_csv(path,
                mode="w" if is_first else "a",
                header=is_first,
                index=False)


def save_local_raw_csv(path: str,
                       data: pd.DataFrame):
        path = os.path.join(
        os.path.expanduser(LOCAL_DATA_PATH),
        "raw",
        f"{path}.csv")
        data.to_csv(path, index=False)


def split_data_csv(raw_dataset_csv: str ):

    csv_path = raw_dataset_csv

    # Assign the input path
    csv_path = os.path.join(
        os.path.expanduser(LOCAL_DATA_PATH),
        "raw",
        f"{csv_path}.csv")
    data = pd.read_csv(csv_path)

    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size=0.3)

    # Create the train dataset output path
    train_path = os.path.join(
        os.path.expanduser(LOCAL_DATA_PATH),
        "raw",
        f"train_{raw_dataset_csv}.csv")

    # Create the validation dataset output path
    val_path = os.path.join(
        os.path.expanduser(LOCAL_DATA_PATH),
        "raw",
        f"val_{raw_dataset_csv}.csv")

    # Output the dataset into csv files
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(val_path, index=False)
