import torch
import cv2
import pandas as pd
import os
import pickle


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, augmentation=None, mode="train"):
        super().__init__()
        self.config = config
        self.metadata = self._load_meta_data()
        self.augmentation = augmentation
        self.mode = mode
        self.data_dict = self._load_data_dict()
        self.n_classes = len(self.id2label)
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row_metadata = self.metadata.iloc[index]

        # load spec. data
        input_spec = self.data_dict[row_metadata.samplename]
        input_spec = cv2.resize(
            input_spec,
            (self.config.IMG_SIZE, self.config.IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # aug
        if self.augmentation is not None:
            input_spec = self.augmentation(image=input_spec)["image"]

        # target
        target = row_metadata.target

        return torch.tensor(input_spec, dtype=torch.float32), torch.tensor(
            target, dtype=torch.long
        )

    def _load_data_dict(self):
        with open(
            os.path.join(self.config.PREPROCESSED_DATA_ROOT, "data_dict.pkl"), "rb"
        ) as f:
            data_dict = pickle.load(f)
        return data_dict

    def _load_meta_data(self):
        metadata_df = pd.read_csv(f"{self.config.DATA_ROOT}/train_metadata.csv")
        train_df = metadata_df[["primary_label", "rating", "filename"]].copy()
        # create target
        train_df["target"] = train_df.primary_label.map(self.label2id)
        # create new sample name
        train_df["samplename"] = train_df.filename.map(
            lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0]
        )
        return train_df

    @property
    def label2id(self):
        # labels
        label_list = sorted(os.listdir(os.path.join(self.config.DATA_ROOT, "train_audio")))
        label_id_list = list(range(len(label_list)))
        label2id = dict(zip(label_list, label_id_list))
        return label2id

    @property
    def id2label(self):
        # labels
        label_list = sorted(os.listdir(os.path.join(self.config.DATA_ROOT, "train_audio")))
        label_id_list = list(range(len(label_list)))
        id2label = dict(zip(label_id_list, label_list))
        return id2label
