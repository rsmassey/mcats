from mcats.ml_logic.params import (COLUMN_NAMES_RAW,
                                            DTYPES_RAW_OPTIMIZED,
                                            DTYPES_RAW_OPTIMIZED_HEADLESS,
                                            DTYPES_PROCESSED_OPTIMIZED
                                            )

from mcats.data_sources.local_disk import (get_pandas_chunk, save_local_chunk)

from mcats.data_sources.big_query import (get_bq_chunk, save_bq_chunk)

import os
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by TO COMPLETE
    """

    return df

def get_chunk(source_name: str,
              index: int = 0,
              chunk_size: int = None,
              verbose=False) -> pd.DataFrame:
    """
    Return a `chunk_size` rows from the source dataset, starting at row `index` (included)
    Always assumes `source_name` (CSV or Big Query table) have headers,
    and do not consider them as part of the data `index` count.
    """

    if "processed" in source_name:
        columns = None
        dtypes = DTYPES_PROCESSED_OPTIMIZED
    else:
        columns = COLUMN_NAMES_RAW
        if os.environ.get("DATA_SOURCE") == "bigquery":
            dtypes = DTYPES_RAW_OPTIMIZED
        else:
            dtypes = DTYPES_RAW_OPTIMIZED_HEADLESS

    if os.environ.get("DATA_SOURCE") == "bigquery":

        chunk_df = get_bq_chunk(table=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                dtypes=dtypes,
                                verbose=verbose)

        return chunk_df

    chunk_df = get_pandas_chunk(path=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                dtypes=dtypes,
                                columns=columns,
                                verbose=verbose)

    return chunk_df


def save_chunk(destination_name: str,
               is_first: bool,
               data: pd.DataFrame) -> None:
    """
    save chunk
    """

    if os.environ.get("DATA_SOURCE") == "bigquery":

        save_bq_chunk(table=destination_name,
                      data=data,
                      is_first=is_first)

        return

    save_local_chunk(path=destination_name,
                     data=data,
                     is_first=is_first)
