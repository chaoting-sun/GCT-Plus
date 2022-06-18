from functools import total_ordering
import os
import dill as pickle
from typing import Tuple

import torch
# from torchtext import data
from torchtext.legacy import data

from Tokenize import moltokenize
from configuration.config_default import LANG_SUPPORTED


def create_fields(lang_format="SMILES",
                  field_path=None, 
                  lang_supported=LANG_SUPPORTED,
                  ) -> Tuple[data.field.Field, data.field.Field]:
    assert lang_format in lang_supported
    
    print("- loading molecule tokenizers...")

    t_src = moltokenize()
    t_trg = moltokenize()

    SRC = data.Field(tokenize=t_src.tokenizer, batch_first=True)
    TRG = data.Field(tokenize=t_trg.tokenizer, batch_first=True,
                     init_token='<sos>', eos_token='<eos>')

    if field_path is not None:
        try:
            print("- loading presaved fields...")
            SRC = pickle.load(open(os.path.join(field_path, 'SRC.pkl'), 'rb'))
            TRG = pickle.load(open(os.path.join(field_path, 'TRG.pkl'), 'rb'))
        except:
            print("- Files SRC.pkl/TRG.pkl not in:" + field_path)
            exit(1)
    return (SRC, TRG)


def condition_fields(cond_list):
    fields = []
    if len(cond_list) > 0:
        for c in cond_list:
            c_field = data.Field(use_vocab=False, sequential=False,
                                 batch_first=True, dtype=torch.float)
            fields.append((c, c_field))
    return fields


def create_total_fields(src_fields, trg_fields, conditions, 
                        fields_fcn=condition_fields):
    total_fields = [('src', src_fields), ('trg', trg_fields)]    
    total_fields.extend(fields_fcn([f'src_{p}' for p in conditions]))
    total_fields.extend(fields_fcn([f'trg_{p}' for p in conditions]))
    return total_fields


def save_fields(src_fields, trg_fields, field_path):
    """ save SRC/TRG fields """
    os.makedirs(field_path, exist_ok=True)
    src_field_path = os.path.join(field_path, 'SRC.pkl')
    trg_field_path = os.path.join(field_path, 'TRG.pkl')

    if os.path.exists(src_field_path):
        exit(f'File already existed: {src_field_path}')
    if os.path.exists(trg_field_path):
        exit(f'File already existed: {trg_field_path}')
    
    pickle.dump(src_fields, open(src_field_path, 'wb'))
    pickle.dump(trg_fields, open(trg_field_path, 'wb'))
