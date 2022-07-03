# from functools import total_ordering
import os
import dill as pickle

import torch
from torchtext import data
# from torchtext.legacy import data

from Tokenize import moltokenize


def smiles_fields(smiles_field_path=None):
    t_src = moltokenize()
    t_trg = moltokenize()

    SRC = data.Field(tokenize=t_src.tokenizer, batch_first=True)
    TRG = data.Field(tokenize=t_trg.tokenizer, batch_first=True,
                     init_token='<sos>', eos_token='<eos>')
    
    if smiles_field_path is not None:
        try:
            SRC = pickle.load(open(os.path.join(smiles_field_path, 'SRC.pkl'), 'rb'))
            TRG = pickle.load(open(os.path.join(smiles_field_path, 'TRG.pkl'), 'rb'))
        except:
            print(">>> Files SRC.pkl/TRG.pkl not in:" + smiles_field_path)
            exit(1)

    return (SRC, TRG)


def condition_fields(conditions):
    return [data.Field(use_vocab=False, sequential=False,
            batch_first=True, dtype=torch.float) for _ in conditions]


# def get_fields(conditions, smiles_field_path=None):
#     SRC, TRG = smiles_fields(smiles_field_path)
#     COND = condition_fields(conditions)
#     total_fields = [('src_no', None), ('trg_no', None),
#                     ('src', SRC), ('trg', TRG)]
#     total_fields.extend([(f'src_{conditions[i]}', COND[i]) for i in range(len(conditions))])
#     total_fields.extend([(f'trg_{conditions[i]}', COND[i]) for i in range(len(conditions))])
#     return total_fields

def get_fields(conditions, smiles_field_path=None):
    # the orders in the DataFrame
    # src,trg,src_no,src_logP,src_tPSA,src_QED,trg_no,trg_logP,trg_tPSA,trg_QED
    SRC, TRG = smiles_fields(smiles_field_path)
    COND = condition_fields(conditions)
    total_fields = [('src', SRC), ('trg', TRG)]
    total_fields.extend([('src_no', None)] + 
                        [(f'src_{conditions[i]}', COND[i]) for i in range(len(conditions))])
    total_fields.extend([('trg_no', None)] +
                        [(f'trg_{conditions[i]}', COND[i]) for i in range(len(conditions))])
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
