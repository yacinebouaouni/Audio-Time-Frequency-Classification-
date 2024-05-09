import torch
import cv2
import numpy as np


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_df, augmentation=None, mode="train"):
        super().__init__()
        self.config = config
        self.data_df = data_df
        self.augmentation = augmentation
        self.mode = mode

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]

        # load spec. data
        input_spec = np.load(sample.filepath)
        input_spec = cv2.resize(
            input_spec,
            (self.config.IMG_SIZE, self.config.IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # aug
        if self.augmentation is not None:
            input_spec = self.augmentation(image=input_spec)["image"]

        # target
        target = sample.target

        return torch.tensor(input_spec, dtype=torch.float32), torch.tensor(
            target, dtype=torch.long
        )
