#%%
import pandas as pd
import numpy as np
import os
import scipy
from scipy import signal

def clean_data(project_path, data_path):

    motion_data_path = os.path.join(data_path, 'motion')
    labels_path = os.path.join(data_path, 'labels')

    # create list to store motion dataframes
    motion_data = []
    for file in os.scandir(motion_data_path):
        print(file.path)
        # use read_csv with whitespace delimiter
        subject_data = pd.read_csv(file.path, delim_whitespace=True, header=None)
        # get subject name from filename
        subject_name = file.name.split('_')[0]
        # add subject name as column
        subject_data['subject'] = subject_name
        # rename columns
        print(subject_data.head())
        subject_data.columns = ['timestamp', 'x', 'y', 'z', 'subject']
        # parse timestamp column as datetime
        subject_data['timestamp'] = pd.to_datetime(subject_data['timestamp'], unit='s')
        print('Motion data head: ', subject_data.head(), ' Motion data minimum: ', subject_data['timestamp'].min(), 'Motion data maximum: ', subject_data['timestamp'].max(), 'Motion data mean: ', subject_data['timestamp'].mean(), sep='\n')
        # add to list
        motion_data.append(subject_data)

    print('came here')
    # create list to store labels dataframes
    labels_data = []
    for file in os.scandir(labels_path):
        print(file.path)
        # get subject name from filename
        subject_data = pd.read_csv(file.path, delim_whitespace=True)
        # get subject name from filename
        subject_name = file.name.split('_')[0]
        # add subject name as column
        subject_data['subject'] = subject_name
        # rename columns
        subject_data.columns = ['timestamp', 'label', 'subject']
        #parse timestamp column as datetime
        subject_data['timestamp'] = pd.to_datetime(subject_data['timestamp'], unit='s')

        #############################
        # This code renames labels to support only wake/ REM/ NREM detecttion

        #new_labels = [0, 1,1,1,5,5,0]

        #############################

        #add to list
        labels_data.append(subject_data)


    #%%
    # Sort both lists by subject ID
    motion_data.sort(key=lambda x: x['subject'].iloc[0])
    labels_data.sort(key=lambda x: x['subject'].iloc[0])
    # Verify lists have same length and that there is 1:1 correspondence between subjects
    print("got here")
    assert len(motion_data) == len(labels_data)
    for i in range(len(motion_data)):
        assert motion_data[i]['subject'].iloc[0] == labels_data[i]['subject'].iloc[0]
        print(motion_data[i]['subject'].iloc[0])



    #%%
    # Now data is cleaned to only keep relevant rows

    # Remove all rows with timestamps less than 0 and greater than the largest labels_data timestamp
    for i in range(len(motion_data)):
        motion_data[i] = motion_data[i][motion_data[i]['timestamp'] >= pd.to_datetime(0)]
        motion_data[i] = motion_data[i][motion_data[i]['timestamp'] <= labels_data[i]['timestamp'].max()]

    # Quantize motion data to 50 Hz, as per paper
    for i in range(len(motion_data)):
        motion_data[i] = motion_data[i].resample('20ms', on='timestamp').mean()
        motion_data[i] = motion_data[i].interpolate(method='linear')
        # resampling makes timestamp an index, thus we make it a column again
        motion_data[i].reset_index(inplace=True)
        print(motion_data[i].head())
        print(' Motion data minimum: ', motion_data[i]['timestamp'].min(), 'Motion data maximum: ', motion_data[i]['timestamp'].max(), 'Motion data mean: ', motion_data[i]['timestamp'].mean(), sep='\n')

    # Downsample labels data to 30 Hz to be compatible with harnet and remove overflow rows if they contain less than 900 samples
    motion_data_downsampled = []
    for i in range(len(motion_data)):
        # downsample labels data using scipy
        motion_data_downsampled.append(motion_data[i][['x', 'y', 'z']].values)
        motion_data_downsampled[i] = signal.resample(motion_data_downsampled[i], int(labels_data[i]['timestamp'].max().timestamp() * 30))
        # ensure length of array is multiple of harnet input size (e.g. 30*30=900)
        nn_input_size = 900

        original_length = len(motion_data_downsampled[i])
        overflow = len(motion_data_downsampled[i]) % nn_input_size
        motion_data_downsampled[i] = (motion_data_downsampled[i][0:-overflow] if overflow != 0 else motion_data_downsampled[i])
        new_length = len(motion_data_downsampled[i])
        print('Original length: ', original_length, 'New length: ', new_length, 'Overflow: ', overflow, sep='\n')


    #%%
    # create one np array containing all the motion data
    for i in range(len(motion_data_downsampled)):
        if i == 0:
            motion_data_downsampled_all = motion_data_downsampled[i]
        else:
            motion_data_downsampled_all = np.concatenate((motion_data_downsampled_all, motion_data_downsampled[i]), axis=0)


    # Create index array that maps indices of motion_data_downsampled_all to subject and label
    index_array = np.zeros((int(len(motion_data_downsampled_all)/nn_input_size), 3), dtype=np.int64)

    index=0
    for i in range(len(motion_data_downsampled)):
        subject_i_length = int(motion_data_downsampled[i].shape[0])
        for j in range(int(subject_i_length/nn_input_size)):
            offset = int(index/nn_input_size)
            index_array[offset, 0] = index
            index_array[offset, 1] = labels_data[i]['label'].iloc[j]
            index_array[offset, 2] = labels_data[i]['subject'].iloc[j]
            index = index + nn_input_size
    # Save data to csv
    index_array_path = os.path.join(data_path, 'index_array.csv')
    motion_data_downsampled_path = os.path.join(data_path, 'motion_data_downsampled_all.csv')

    pd.DataFrame(index_array, columns=['index', 'label', 'subject']).to_csv(index_array_path, index=False)
    pd.DataFrame(motion_data_downsampled_all, columns=['x', 'y', 'z']).to_csv(motion_data_downsampled_path, index=False)