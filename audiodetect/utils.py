import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch import nn
import pandas as pd
from audiodetect.data import AudioDataset
from audiodetect.models import BirdModel
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import pytorch_lightning as pl

from audiodetect.metric import score
from sklearn.model_selection import KFold
import os
import json


def get_labels(config):
    label2id = json.load(open(os.path.join(config.DATA_ROOT, "label2id.json")))
    label_list = sorted(os.listdir(os.path.join(config.DATA_ROOT, "train_audio")))
    n_classes = len(label2id)
    return (label_list, label2id, n_classes)


def prepare_data(config, label2id):
    data_df = pd.read_csv(f"{config.DATA_ROOT}/train_metadata.csv")
    data_df = data_df[["primary_label", "rating", "filename"]].copy()
    data_df["target"] = data_df.primary_label.map(label2id)
    data_df["filepath"] = (
        config.PREPROCESSED_DATA_ROOT
        + "/train_audio/"
        + data_df.filename.str.replace("ogg", "npy")
    )
    data_df["samplename"] = data_df.filename.map(
        lambda x: x.split("/")[0] + "-" + x.split("/")[-1].split(".")[0]
    )

    # K-Fold
    kf = KFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED)
    data_df["fold"] = 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_df)):
        data_df.loc[val_idx, "fold"] = fold
    return data_df


def train(fold_id, total_df, config, label_list, n_classes):
    print("================================================================")
    print(f"==== Running training for fold {fold_id} ====")

    # == create dataset and dataloader ==
    train_df = total_df[total_df["fold"] != fold_id].copy()
    valid_df = total_df[total_df["fold"] == fold_id].copy()

    print(f"Train Samples: {len(train_df)}")
    print(f"Valid Samples: {len(valid_df)}")

    train_ds = AudioDataset(config, train_df, None, "train")
    val_ds = AudioDataset(config, valid_df, None, "valid")

    train_dl = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # == init model ==
    bird_model = BirdModel(config=config, label_list=label_list)

    # == init callback ==
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        dirpath=config.OUTPUT_DIR,
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        filename=f"fold_{fold_id}",
        mode="max",
    )
    callbacks_to_use = [checkpoint_callback, TQDMProgressBar(refresh_rate=1)]

    # == init trainer ==
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        val_check_interval=0.5,
        callbacks=callbacks_to_use,
        enable_model_summary=False,
        accelerator="gpu",
        deterministic=True,
        precision="16-mixed" if config.MIXED_PRECISION else 32,
    )

    # == Training ==
    trainer.fit(bird_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # == Prediction ==
    best_model_path = checkpoint_callback.best_model_path
    weights = torch.load(best_model_path)["state_dict"]
    bird_model.load_state_dict(weights)

    preds, gts = predict(config, val_dl, bird_model, n_classes)

    # = create dataframe =
    pred_df = pd.DataFrame(preds, columns=label_list)
    pred_df["id"] = np.arange(len(pred_df))
    gt_df = pd.DataFrame(gts, columns=label_list)
    gt_df["id"] = np.arange(len(gt_df))

    # = compute score =
    val_score = score(gt_df, pred_df, row_id_column_name="id")

    # == save to file ==
    if config.SAVE_VALID_PREDS:
        pred_cols = [f"pred_{t}" for t in label_list]
        valid_df = pd.concat(
            [
                valid_df.reset_index(),
                pd.DataFrame(
                    np.zeros((len(valid_df), len(label_list) * 2)).astype(np.float32),
                    columns=label_list + pred_cols,
                ),
            ],
            axis=1,
        )
        valid_df[label_list] = gts
        valid_df[pred_cols] = preds
        valid_df.to_csv(f"{config.OUTPUT_DIR}/pred_df_f{fold_id}.csv", index=False)

    return preds, gts, val_score


def predict(config, data_loader, model, n_classes):
    model.to(config.DEVICE)
    model.eval()
    predictions = []
    gts = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            x, y = batch
            x = x.cuda()
            outputs = model(x)
            outputs = nn.Softmax(dim=1)(outputs)
        predictions.append(outputs.detach().cpu())
        gts.append(y.detach().cpu())

    predictions = torch.cat(predictions, dim=0).cpu().detach()
    gts = torch.cat(gts, dim=0).cpu().detach()
    gts = torch.nn.functional.one_hot(gts, n_classes)

    return predictions.numpy().astype(np.float32), gts.numpy().astype(np.float32)
