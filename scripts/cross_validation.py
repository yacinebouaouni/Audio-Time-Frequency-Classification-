from omegaconf import OmegaConf
from sklearn.model_selection import KFold

from audiodetect.data.dataset import AudioDataset
import torch
import pandas as pd

import json 
import os
import numpy as np

import pytorch_lightning as pl
from audiodetect.models.trainer import BirdModel
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from audiodetect.metric import score
from audiodetect.utils import predict

torch.set_float32_matmul_precision('high')


def run_training(fold_id, total_df):
    print('================================================================')
    print(f"==== Running training for fold {fold_id} ====")
    
    # == create dataset and dataloader ==
    train_df = total_df[total_df['fold'] != fold_id].copy()
    valid_df = total_df[total_df['fold'] == fold_id].copy()
    
    print(f'Train Samples: {len(train_df)}')
    print(f'Valid Samples: {len(valid_df)}')
    

    train_ds = AudioDataset(config, train_df, None, 'train')
    val_ds = AudioDataset(config, valid_df, None, 'valid')
     
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    # == init model ==
    bird_model = BirdModel(config=config, label_list=label_list)
    
    # == init callback ==
    checkpoint_callback = ModelCheckpoint(monitor='val_score',
                                          dirpath=config.OUTPUT_DIR,
                                          save_top_k=1,
                                          save_last=False,
                                          save_weights_only=True,
                                          filename=f"fold_{fold_id}",
                                          mode='max')
    callbacks_to_use = [checkpoint_callback, TQDMProgressBar(refresh_rate=1)]
    
    # == init trainer ==
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        val_check_interval=0.5,
        callbacks=callbacks_to_use,
        enable_model_summary=False,
        accelerator="gpu",
        deterministic=True,
        precision='16-mixed' if config.MIXED_PRECISION else 32,
    )
    
    # == Training ==
    trainer.fit(bird_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    # == Prediction ==
    best_model_path = checkpoint_callback.best_model_path
    weights = torch.load(best_model_path)['state_dict']
    bird_model.load_state_dict(weights)
    
    preds, gts = predict(config, val_dl, bird_model, n_classes)
    
    # = create dataframe =
    pred_df = pd.DataFrame(preds, columns=label_list)
    pred_df['id'] = np.arange(len(pred_df))
    gt_df = pd.DataFrame(gts, columns=label_list)
    gt_df['id'] = np.arange(len(gt_df))
    
    # = compute score =
    val_score = score(gt_df, pred_df, row_id_column_name='id')
    
    # == save to file ==
    pred_cols = [f'pred_{t}' for t in label_list]
    valid_df = pd.concat([valid_df.reset_index(), pd.DataFrame(np.zeros((len(valid_df), len(label_list)*2)).astype(np.float32), columns=label_list+pred_cols)], axis=1)
    valid_df[label_list] = gts
    valid_df[pred_cols] = preds
    valid_df.to_csv(f"{config.OUTPUT_DIR}/pred_df_f{fold_id}.csv", index=False)
    
    return preds, gts, val_score



# == prepare data ==
config = OmegaConf.load('../config/config.yaml')
data_df = pd.read_csv(f'{config.DATA_ROOT}/train_metadata.csv')
train_df = data_df[['primary_label', 'rating', 'filename']].copy()
label2id = json.load(open(os.path.join(config.DATA_ROOT, "label2id.json")))
label_list = sorted(os.listdir(os.path.join(config.DATA_ROOT, 'train_audio')))
label_id_list = list(range(len(label_list)))
n_classes = len(label2id)
train_df['target'] = train_df.primary_label.map(label2id)
train_df['filepath'] = config.PREPROCESSED_DATA_ROOT + '/train_audio/' + train_df.filename.str.replace('ogg', 'npy')
train_df['samplename'] = train_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    
kf = KFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED)
train_df['fold'] = 0
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, 'fold'] = fold
    
    
# record
fold_val_score_list = list()
oof_df = train_df.copy()
pred_cols = [f'pred_{t}' for t in label_list]
oof_df = pd.concat([oof_df, pd.DataFrame(np.zeros((len(oof_df), len(pred_cols)*2)).astype(np.float32), columns=label_list+pred_cols)], axis=1)

for f in range(config.FOLDS):
    
    # get validation index
    val_idx = list(train_df[train_df['fold'] == f].index)
    
    print(len(train_df))
    # main loop of f-fold
    val_preds, val_gts, val_score = run_training(f, train_df)
    
    # record
    oof_df.loc[val_idx, label_list] = val_gts
    oof_df.loc[val_idx, pred_cols] = val_preds
    fold_val_score_list.append(val_score)
    
    # only training one fold
    break


for idx, val_score in enumerate(fold_val_score_list):
    print(f'Fold {idx} Val Score: {val_score:.5f}')


oof_df.to_csv(f"{config.OUTPUT_DIR}/oof_pred.csv", index=False)