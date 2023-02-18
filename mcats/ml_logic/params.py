"""
mcats model package params
load and validate the environment variables in the `.env`
"""

import os
import numpy as np
a = os.environ

LOCAL_SONG_PATH = os.path.join(os.path.expanduser('~'),
                               "code",
                               "rsmassey",
                               "mcats",
                               "mcats",
                               "data",
                               "song_input.wav")

DATASET_SIZE = os.environ.get("DATASET_SIZE")
VALIDATION_DATASET_SIZE = os.environ.get("VALIDATION_DATASET_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
PROJECT = os.environ.get("PROJECT")
DATASET = os.environ.get("DATASET")

# Use this to optimize loading of raw_data with headers: pd.read_csv(..., dtypes=..., header=True)
DTYPES_RAW_OPTIMIZED = {
                    "tempo":"float32",
                    "beats_mean":"float32",
                    "beats_var":"float32",
                    "zero_crossings_mean":"float32",
                    "zero_crossings_var":"float32",
                    "spectral_centroids_mean":"float32",
                    "spectral_centroids_var":"float32",
                    "spectral_rolloff_mean":"float32",
                    "spectral_rolloff_var":"float32",
                    "mfcc_1_mean":"float32",
                    "mfcc_1_var":"float32",
                    "mfcc_2_mean":"float32",
                    "mfcc_2_var":"float32",
                    "mfcc_3_mean":"float32",
                    "mfcc_3_var":"float32",
                    "mfcc_4_mean":"float32",
                    "mfcc_4_var":"float32",
                    "mfcc_5_mean":"float32",
                    "mfcc_5_var":"float32",
                    "mfcc_6_mean":"float32",
                    "mfcc_6_var":"float32",
                    "mfcc_7_mean":"float32",
                    "mfcc_7_var":"float32",
                    "mfcc_8_mean":"float32",
                    "mfcc_8_var":"float32",
                    "mfcc_9_mean":"float32",
                    "mfcc_9_var":"float32",
                    "mfcc_10_mean":"float32",
                    "mfcc_10_var":"float32",
                    "mfcc_11_mean":"float32",
                    "mfcc_11_var":"float32",
                    "mfcc_12_mean":"float32",
                    "mfcc_12_var":"float32",
                    "mfcc_13_mean":"float32",
                    "mfcc_13_var":"float32",
                    "mfcc_14_mean":"float32",
                    "mfcc_14_var":"float32",
                    "mfcc_15_mean":"float32",
                    "mfcc_15_var":"float32",
                    "mfcc_16_mean":"float32",
                    "mfcc_16_var":"float32",
                    "mfcc_17_mean":"float32",
                    "mfcc_17_var":"float32",
                    "mfcc_18_mean":"float32",
                    "mfcc_18_var":"float32",
                    "mfcc_19_mean":"float32",
                    "mfcc_19_var":"float32",
                    "mfcc_20_mean":"float32",
                    "mfcc_20_var":"float32",
                    "mfcc_21_mean":"float32",
                    "mfcc_21_var":"float32",
                    "mfcc_22_mean":"float32",
                    "mfcc_22_var":"float32",
                    "mfcc_23_mean":"float32",
                    "mfcc_23_var":"float32",
                    "mfcc_24_mean":"float32",
                    "mfcc_24_var":"float32",
                    "mfcc_25_mean":"float32",
                    "mfcc_25_var":"float32",
                    "mfcc_26_mean":"float32",
                    "mfcc_26_var":"float32",
                    "mfcc_27_mean":"float32",
                    "mfcc_27_var":"float32",
                    "mfcc_28_mean":"float32",
                    "mfcc_28_var":"float32",
                    "mfcc_29_mean":"float32",
                    "mfcc_29_var":"float32",
                    "mfcc_30_mean":"float32",
                    "mfcc_30_var":"float32",
                    "mfcc_31_mean":"float32",
                    "mfcc_31_var":"float32",
                    "mfcc_32_mean":"float32",
                    "mfcc_32_var":"float32",
                    "mfcc_33_mean":"float32",
                    "mfcc_33_var":"float32",
                    "mfcc_34_mean":"float32",
                    "mfcc_34_var":"float32",
                    "mfcc_35_mean":"float32",
                    "mfcc_35_var":"float32",
                    "mfcc_36_mean":"float32",
                    "mfcc_36_var":"float32",
                    "mfcc_37_mean":"float32",
                    "mfcc_37_var":"float32",
                    "mfcc_38_mean":"float32",
                    "mfcc_38_var":"float32",
                    "mfcc_39_mean":"float32",
                    "mfcc_39_var":"float32",
                    "mfcc_40_mean":"float32",
                    "mfcc_40_var":"float32",
                    "genre":"float32"
}

COLUMN_NAMES_RAW = DTYPES_RAW_OPTIMIZED.keys()

# Use this to optimize loading of raw_data without headers: pd.read_csv(..., dtypes=..., header=False)
DTYPES_RAW_OPTIMIZED_HEADLESS = {
                    0:"float32",
                    1:"float32",
                    2:"float32",
                    3:"float32",
                    4:"float32",
                    5:"float32",
                    6:"float32",
                    7:"float32",
                    8:"float32",
                    9:"float32",
                    10:"float32",
                    11:"float32",
                    12:"float32",
                    13:"float32",
                    14:"float32",
                    15:"float32",
                    16:"float32",
                    17:"float32",
                    18:"float32",
                    19:"float32",
                    20:"float32",
                    21:"float32",
                    22:"float32",
                    23:"float32",
                    24:"float32",
                    25:"float32",
                    26:"float32",
                    27:"float32",
                    28:"float32",
                    29:"float32",
                    30:"float32",
                    31:"float32",
                    32:"float32",
                    33:"float32",
                    34:"float32",
                    35:"float32",
                    36:"float32",
                    37:"float32",
                    38:"float32",
                    39:"float32",
                    40:"float32",
                    41:"float32",
                    42:"float32",
                    43:"float32",
                    44:"float32",
                    45:"float32",
                    46:"float32",
                    47:"float32",
                    48:"float32",
                    49:"float32",
                    50:"float32",
                    51:"float32",
                    52:"float32",
                    53:"float32",
                    54:"float32",
                    55:"float32",
                    56:"float32",
                    57:"float32",
                    58:"float32",
                    59:"float32",
                    60:"float32",
                    61:"float32",
                    62:"float32",
                    63:"float32",
                    64:"float32",
                    65:"float32",
                    66:"float32",
                    67:"float32",
                    68:"float32",
                    69:"float32",
                    70:"float32",
                    71:"float32",
                    72:"float32",
                    73:"float32",
                    74:"float32",
                    75:"float32",
                    76:"float32",
                    77:"float32",
                    78:"float32",
                    79:"float32",
                    80:"float32",
                    81:"float32",
                    82:"float32",
                    83:"float32",
                    84:"float32",
                    85:"float32",
                    86:"float32",
                    87:"float32",
                    88:"float32",
                    89:"float32"
}

DTYPES_PROCESSED_OPTIMIZED = np.float32



################## VALIDATIONS #################

env_valid_options = dict(
    DATASET_SIZE=["small_30s"],
    VALIDATION_DATASET_SIZE=["small_30s"],
    DATA_SOURCE=["local", "bigquery"],
    MODEL_TARGET=["local", "gcs", "mlflow"],
    PREFECT_BACKEND=["development", "production"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
