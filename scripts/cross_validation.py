"""
Module for training a deep learning model for audio classification.

This module provides functionality for training a deep learning model for audio classification using the specified configuration settings.

Functions:
- load_arguments(): Parses command-line arguments for training.
- main(config): Main function for training the model.

Example Usage:
    $ python train.py --lr 0.001 --single_fold
    This command will train the model with a learning rate of 0.001 for one single fold.
"""

from omegaconf import OmegaConf

import torch
import pandas as pd
import numpy as np

from audiodetect.utils import train, get_labels, prepare_data

import argparse

torch.set_float32_matmul_precision("high")


def load_arguments():
    parser = argparse.ArgumentParser(description="Load arguments for training")
    parser.add_argument(
        "--lr", type=float, required=True, help="Learning rate for training"
    )
    parser.add_argument(
        "--single_fold",
        action="store_true",
        help="Set this flag to train/validate on single fold.",
    )
    args = parser.parse_args()
    return args


def main(config):
    (label_list, label2id, n_classes) = get_labels(config)
    data_df = prepare_data(config=config, label2id=label2id)

    # record
    fold_val_score_list = list()
    oof_df = data_df.copy()
    pred_cols = [f"pred_{t}" for t in label_list]
    oof_df = pd.concat(
        [
            oof_df,
            pd.DataFrame(
                np.zeros((len(oof_df), len(pred_cols) * 2)).astype(np.float32),
                columns=label_list + pred_cols,
            ),
        ],
        axis=1,
    )

    for f in range(config.FOLDS):
        # get validation index
        val_idx = list(data_df[data_df["fold"] == f].index)

        # main loop of f-fold
        val_preds, val_gts, val_score = train(f, data_df, config, label_list, n_classes)

        # record
        oof_df.loc[val_idx, label_list] = val_gts
        oof_df.loc[val_idx, pred_cols] = val_preds
        fold_val_score_list.append(val_score)
        if SINGLE_FOLD:
            break

    for idx, val_score in enumerate(fold_val_score_list):
        print(f"Fold {idx} Val Score: {val_score:.5f}")


if __name__ == "__main__":
    args = load_arguments()
    config = OmegaConf.load("../config/config.yaml")
    config.LR = args.lr
    SINGLE_FOLD = args.single_fold
    main(config)
