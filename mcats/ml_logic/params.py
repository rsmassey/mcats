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
                "length":"int32",
                "chroma_stft_mean":"float32",
                "chroma_stft_var":"float32",
                "rms_mean":"float32",
                "rms_var":"float32",
                "spectral_centroid_mean":"float32",
                "spectral_centroid_var":"float32",
                "spectral_bandwidth_mean":"float32",
                "spectral_bandwidth_var":"float32",
                "rolloff_mean":"float32",
                "rolloff_var":"float32",
                "zero_crossing_rate_mean":"float32",
                "zero_crossing_rate_var":"float32",
                "harmony_mean":"float32",
                "harmony_var":"float32",
                "perceptr_mean":"float32",
                "perceptr_var":"float32",
                "tempo":"float32",
                "mfcc1_mean":"float32",
                "mfcc1_var":"float32",
                "mfcc2_mean":"float32",
                "mfcc2_var":"float32",
                "mfcc3_mean":"float32",
                "mfcc3_var":"float32",
                "mfcc4_mean":"float32",
                "mfcc4_var":"float32",
                "mfcc5_mean":"float32",
                "mfcc5_var":"float32",
                "mfcc6_mean":"float32",
                "mfcc6_var":"float32",
                "mfcc7_mean":"float32",
                "mfcc7_var":"float32",
                "mfcc8_mean":"float32",
                "mfcc8_var":"float32",
                "mfcc9_mean":"float32",
                "mfcc9_var":"float32",
                "mfcc10_mean":"float32",
                "mfcc10_var":"float32",
                "mfcc11_mean":"float32",
                "mfcc11_var":"float32",
                "mfcc12_mean":"float32",
                "mfcc12_var":"float32",
                "mfcc13_mean":"float32",
                "mfcc13_var":"float32",
                "mfcc14_mean":"float32",
                "mfcc14_var":"float32",
                "mfcc15_mean":"float32",
                "mfcc15_var":"float32",
                "mfcc16_mean":"float32",
                "mfcc16_var":"float32",
                "mfcc17_mean":"float32",
                "mfcc17_var":"float32",
                "mfcc18_mean":"float32",
                "mfcc18_var":"float32",
                "mfcc19_mean":"float32",
                "mfcc19_var":"float32",
                "mfcc20_mean":"float32",
                "mfcc20_var":"float32",
                "label":"string"
}

COLUMN_NAMES_RAW = DTYPES_RAW_OPTIMIZED.keys()

# Use this to optimize loading of raw_data without headers: pd.read_csv(..., dtypes=..., header=False)
DTYPES_RAW_OPTIMIZED_HEADLESS = {
                    0:"int32",
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
                    58:"string"
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
