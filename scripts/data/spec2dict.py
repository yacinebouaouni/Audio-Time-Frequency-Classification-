import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os
import pickle 

def load_data_as_dict(df, filepath_column, samplename_column):
    """
    Loads data from file paths in a dataframe into a dictionary.

    Parameters:
    - df: pandas DataFrame containing file paths and sample names
    - filepath_column: name of the column containing file paths
    - samplename_column: name of the column containing sample names

    Returns:
    - data_dict: dictionary where keys are sample names and values are numpy arrays loaded from file paths
    """
    data_dict = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Data"):
        filepath = row[filepath_column]
        samplename = row[samplename_column]
        array_data = np.load(filepath)
        data_dict[samplename] = array_data

    return data_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument(
        "--root_path", type=str, required=True, help="Path to the root directory of the metadata file."
    )
    parser.add_argument(
        "--dst_path",
        required=True,
        type=str,
        help="Path to the destination directory for processed spectrograms dict.",
    )
    return parser.parse_args()



if __name__=="__main__":
    
    args = parse_arguments()
    metadata_df = pd.read_csv(f'{args.root_path}/train_metadata.csv')
    train_df = metadata_df[['primary_label', 'rating', 'filename']].copy()

    # get filepath to sepctrograms
    train_df['filepath']   = args.dst_path + '/train_audio/' + train_df.filename.str.replace('ogg', 'npy')
    train_df['samplename'] = train_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    print(f'find {len(train_df)} samples')
    
    # get dictionary of spectrograms
    data_dict = load_data_as_dict(train_df, "filepath", "samplename")
    # Save the dictionary using pickle
    file_path = os.path.join(args.dst_path, "spectrograms_dict.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)    
    

