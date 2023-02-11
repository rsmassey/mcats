"""
mcats model package params
load and validate the environment variables in the `.env`
"""

import os

LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'),
                               "code",
                               "rsmassey",
                               "mcats",
                               "mcats",
                               "data",
                               "normalized_data.csv")

LOCAL_SONG_PATH = os.path.join(os.path.expanduser('~'),
                               "code",
                               "rsmassey",
                               "mcats",
                               "mcats",
                               "data",
                               "song_input.wav")
