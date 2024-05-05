from audiodetect.data import AudioDataset
from omegaconf import OmegaConf

config = OmegaConf.load('../config/config.yaml')
dataset = AudioDataset(config=config)

print(len(dataset))
print(dataset.id2label)
print(dataset.label2id)
print(dataset[0])