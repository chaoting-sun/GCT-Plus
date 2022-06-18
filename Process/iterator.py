import os
import joblib
import argparse
import pandas as pd
import dill as pickle
from typing import List

from torchtext import data
# from torchtext.legacy import data
from sklearn.preprocessing import RobustScaler, StandardScaler

import Process.batch as bt
from configuration.config_default import CONDITIONS

SEED = 42
SPLIT_RATIO = 0.8


def create_iterator(opt: argparse.Namespace,
                    tr_te: str,
                    source: List[str],
                    target: List[str],
                    SRC: data.field.Field,
                    TRG: data.field.Field,
                    condition: pd.DataFrame,
                    debug: bool
                    ) -> bt.MyIterator:

    df = pd.DataFrame({
        'src': [line for line in source],
        'trg': [line for line in target]},
        columns=["src", "trg"]
    )

    # correct the condition order
    condition = condition.reindex(columns=CONDITIONS)

    # get condition scaler
    if opt.scaler_folder is not None:
        scaler_path = os.path.join(opt.scaler_folder, 'scaler.pkl')
    else:
        scaler_path = None
    scaler = get_scaler(condition, scaler_path=scaler_path)

    # transform by scaler
    condition = scaler_transform(condition, scaler)

    df = pd.concat([df, condition], axis=1)
    df = mask_invalid_len_data(df, opt.nconds,
                               opt.lang_format, opt.max_strlen)

    data_fields = [('src', SRC), ('trg', TRG)]
    data_fields = extend_fields(data_fields, opt.cond_list)
    data_path = os.path.join(opt.data_path, 'DB_temp.csv')
    df.to_csv(data_path, index=False)

    dataset = data.TabularDataset(data_path, format='csv',
                                  fields=data_fields, skip_header=True)

    if tr_te == "train":
        toklenList = []
        for i in range(len(dataset)):
            toklenList.append(len(vars(dataset[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv(os.path.join(
            opt.data_path, "toklen_list.csv"), index=False)

    data_iter = get_iterator(dataset, opt.batch_size,
                             opt.device, data_type=tr_te, debug=debug)

    # print(" - dict-key:", dataset[0].__dict__.keys())
    # print(" - source:", dataset[0].src)

    if tr_te == "train":
        if opt.load_field is False:
            print(" - building vocab from train data...")
            SRC.build_vocab(dataset)
            TRG.build_vocab(dataset)

            field_folder = os.path.join(opt.data_path, opt.field_path)
            os.makedirs(field_folder, exist_ok=True)
            pickle.dump(SRC, open(os.path.join(field_folder, 'SRC.pkl'), 'wb'))
            pickle.dump(TRG, open(os.path.join(field_folder, 'TRG.pkl'), 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']
        assert opt.src_pad == opt.trg_pad

        opt.train_len = sum(1 for _ in data_iter)

    elif tr_te == "test":
        opt.test_len = sum(1 for _ in data_iter)

    return data_iter


def create_aug_iterator(opt: argparse.Namespace,
                        tr_te: str,
                        source: List[str],
                        target: List[str],
                        SRC: data.field.Field,
                        TRG: data.field.Field,
                        src_conds: pd.DataFrame,
                        trg_conds: pd.DataFrame,
                        debug: bool
                        ) -> bt.MyIterator:

    df = pd.DataFrame({
        'src': [line for line in source],
        'trg': [line for line in target]},
        columns=["src", "trg"]
    )

    # check if the property orders of the data and the scaler are match
    # the scaler order from molGCT: [[logP, tPSA, QED]]
    if list(conds.columns) != CONDITIONS:
        conds = conds.reindex(columns=CONDITIONS)

    # Get the condition scaler
    if not opt.scaler_folder:
        scaler_path = os.path.join(opt.scaler_folder, 'scaler.pkl')
        try:
            scaler = joblib.load(scaler_path)
            print("- load scaler from", scaler_path)
        except:
            exit(f"error: {scaler_path} file not found")
    else:
        scaler = RobustScaler(quantile_range=(0.1, 0.9))
        scaler.fit(src_conds.copy(), len(src_conds.columns))
        joblib.dump(scaler, open(os.path.join(
            scaler_path, 'new_scaler.pkl'), 'wb'))

        print("- create map scaler")

    # transform conditions
    src_conds = transform_conditions(src_conds, scaler)
    trg_conds = transform_conditions(trg_conds, scaler)

    # change condition name to distinguish src & trg
    src_conds.rename(columns={f"src_{p}" for p in CONDITIONS}, inplace=True)
    trg_conds.rename(columns={f"trg_{p}" for p in CONDITIONS}, inplace=True)

    df = pd.concat([df, src_conds, trg_conds], axis=1)
    df = mask_invalid_len_data(df, opt.nconds,
                               opt.lang_format, opt.max_strlen)

    data_fields = [('src', SRC), ('trg', TRG)]
    cond_list = []
    cond_list.extend([f'src_{p}' for p in opt.cond_list])
    cond_list.extend([f'trg_{p}' for p in opt.cond_list])

    data_fields = extend_fields(data_fields, opt.cond_list)
    data_path = os.path.join(opt.data_path, 'DB_temp.csv')
    df.to_csv(data_path, index=False)

    dataset = data.TabularDataset(data_path, format='csv',
                                  fields=data_fields, skip_header=True)

    if tr_te == "train":
        toklenList = []
        for i in range(len(dataset)):
            toklenList.append(len(vars(dataset[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv(os.path.join(
            opt.data_path, "toklen_list.csv"), index=False)

    data_iter = get_iterator(dataset, opt.batch_size,
                             opt.device, data_type=tr_te, debug=debug)

    # print(" - dict-key:", dataset[0].__dict__.keys())
    # print(" - source:", dataset[0].src)

    if tr_te == "train":
        if opt.load_field is False:
            print(" - building vocab from train data...")
            SRC.build_vocab(dataset)
            TRG.build_vocab(dataset)

            field_folder = os.path.join(opt.data_path, opt.field_path)
            os.makedirs(field_folder, exist_ok=True)
            pickle.dump(SRC, open(os.path.join(field_folder, 'SRC.pkl'), 'wb'))
            pickle.dump(TRG, open(os.path.join(field_folder, 'TRG.pkl'), 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']
        assert opt.src_pad == opt.trg_pad

        opt.train_len = sum(1 for _ in data_iter)

    elif tr_te == "test":
        opt.test_len = sum(1 for _ in data_iter)

    return data_iter
