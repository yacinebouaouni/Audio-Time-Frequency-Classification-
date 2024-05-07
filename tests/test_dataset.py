from audiodetect.data import AudioDataset
from omegaconf import OmegaConf
import pandas as pd
import os
import json

config = OmegaConf.load('../config/config.yaml')
data_df = pd.read_csv(f'{config.DATA_ROOT}/train_metadata.csv')
train_df = data_df[['primary_label', 'rating', 'filename']].copy()
label2id = json.load(open(os.path.join(config.DATA_ROOT, "label2id.json")))


train_df['target'] = train_df.primary_label.map(label2id)
train_df['filepath'] = config.PREPROCESSED_DATA_ROOT + '/train_audio/' + train_df.filename.str.replace('ogg', 'npy')
train_df['samplename'] = train_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

dataset = AudioDataset(config=config, data_df=train_df)

print(len(dataset))
print(dataset[0])