import numpy as np
import pandas as pd
import os

from logistic_model import train_and_classify
import audio_to_df
import feature_extraction
import file_to_prediction


if __name__ == '__main__':
    try:
        preprocess_and_train()
        train_and_classify()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
