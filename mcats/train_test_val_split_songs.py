def train_test_val_split_songs(json_dir):

    all_songs = []

    train_songs = []
    val_songs = []
    test_songs = []

    X_train = []
    X_val = []
    y_train = []
    y_val = []
    X_test = []
    y_test = []
    target_shape = (20, 130)
    segments = 10

    # List of file names to be merged
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    # Iterate through each file name and load the names of all songs into a list
    for file_name in json_files:
        with open(file_name, 'r') as file:
            data = json.load(file)
            song_names = list(data.keys())

        for song in song_names:
            if song not in all_songs:
                all_songs.append(song)

    train_songs, test_songs = train_test_split(all_songs, test_size=0.20, random_state=42)
    train_songs, val_songs = train_test_split(train_songs, test_size=0.25, random_state=42)


    # Iterate through each file name and load the JSON data
    for file_name in json_files:
        with open(file_name, 'r') as file:
            data = json.load(file)

        # Iterate over each song and extract the MFCCs
        for song_name in list(data.keys()):
            if 'seg_0' in data[song_name]:
                for i in range(segments):
                    # Create a NumPy array of all MFCCs for each segment
                    mfcc_array = np.array(data[song_name][f'seg_{i}'])
                    pad_width = [(0, max(0, target_shape[i] - mfcc_array.shape[i])) for i in range(len(target_shape))]
                    padded_mfcc = np.pad(mfcc_array, pad_width=pad_width, mode='constant', constant_values=0)

                    if song_name in train_songs:
                        X_train.append(padded_mfcc)
                        y_train.append(data[song_name]['genre'])
                    elif song_name in val_songs:
                        X_val.append(padded_mfcc)
                        y_val.append(data[song_name]['genre'])
                    elif song_name in test_songs:
                        X_test.append(padded_mfcc)
                        y_test.append(data[song_name]['genre'])

    # Convert the lists to NumPy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_val, y_train, y_val, X_test, y_test
