# from functools import total_ordering
import os
import dill as pickle

import torch
from torchtext import data
# from torchtext.legacy import data
from SmilesPE.pretokenizer import atomwise_tokenizer


class moltokenize(object):
    def tokenizer(self, sentence):
        return [tok for tok in atomwise_tokenizer(sentence) if tok != " "]

    @staticmethod
    def untokenizer(tokens, sos_idx, eos_idx, itos):
        smi = ""
        for token in tokens:
            if token == eos_idx:
                break
            elif token != sos_idx:
                smi += itos[token]
        return smi


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

# torch.float32 is equal to torch.float
def condition_fields(conditions):
    return [data.Field(use_vocab=False, sequential=False,
            batch_first=True, dtype=torch.float32) for _ in conditions]


# def get_fields(conditions, smiles_field_path=None):
#     SRC, TRG = smiles_fields(smiles_field_path)
#     COND = condition_fields(conditions)
#     total_fields = [('src_no', None), ('trg_no', None),
#                     ('src', SRC), ('trg', TRG)]
#     total_fields.extend([(f'src_{conditions[i]}', COND[i]) for i in range(len(conditions))])
#     total_fields.extend([(f'trg_{conditions[i]}', COND[i]) for i in range(len(conditions))])
#     return total_fields


def get_cvaetfencoder_fields(conditions, smiles_field_path=None):
    # the orders in the DataFrame
    # src,trg,src_no,src_logP,src_tPSA,src_QED,trg_no,trg_logP,trg_tPSA,trg_QED
    SRC, TRG = smiles_fields(smiles_field_path)
    COND = condition_fields(conditions)
    total_fields = [('src', SRC), ('trg_en', SRC), ('trg', TRG)]
    total_fields.extend([('src_no', None)] +
                        [(f'src_{conditions[i]}', COND[i]) for i in range(len(conditions))])
    total_fields.extend([('trg_no', None)] +
                        [(f'trg_{conditions[i]}', COND[i]) for i in range(len(conditions))])
    return total_fields, SRC, TRG


def get_tf_fields(conditions, smiles_field_path=None):
    """
    order: src_no,src,src_logP,src_tPSA,src_QED,trg_no,trg,trg_logP,trg_tPSA,trg_QE
    """
    SRC, TRG = smiles_fields(smiles_field_path)
    COND = condition_fields(conditions)
    
    src_fields = [('src_no', None), ('src', SRC)]
    for i in range(len(conditions)):
        src_fields.append((f'src_{conditions[i]}', COND[i]))
    trg_fields = [('trg_no', None), ('trg', TRG)]
    for i in range(len(conditions)):
        trg_fields.append((f'trg_{conditions[i]}', COND[i]))
    total_fields = src_fields + trg_fields
    return total_fields, SRC, TRG


def get_inference_fields(conds, smiles_fields_path):
    SRC, TRG = smiles_fields(smiles_fields_path)
    PROP = condition_fields(conds)
    fsmiles = [('src', SRC)]
    fconds = [(f'src_{conds[i]}', PROP[i])
               for i in range(len(conds))] + \
              [(f'trg_{conds[i]}', PROP[i])
               for i in range(len(conds))]
    return fsmiles + fconds, SRC, TRG


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


<<<<<<< HEAD
def smiles_field(properties, field_path=None, suffix=None):
    t_src = moltokenize()
    t_trg = moltokenize()
    
    SRC = data.Field(
        tokenize=t_src.tokenizer,
        batch_first=True,
        unk_token='<unk>'
    )
    TRG = data.Field(
        tokenize=t_trg.tokenizer,
        batch_first=True,
        init_token='<sos>',
        eos_token='<eos>',
        unk_token='<unk>'
    )
    
    if field_path is not None:
        if suffix is None:
            suffix = '-'.join(properties)
        try:
            SRC = pickle.load(open(os.path.join(field_path, f'SRC_{suffix}.pkl'), 'rb'))
            TRG = pickle.load(open(os.path.join(field_path, f'TRG_{suffix}.pkl'), 'rb'))
=======
def smiles_field(properties, smiles_field_path=None):
    t_src = moltokenize()
    t_trg = moltokenize()
    
    SRC = data.Field(
        tokenize=t_src.tokenizer,
        batch_first=True,
        unk_token='<unk>'
    )
    TRG = data.Field(
        tokenize=t_trg.tokenizer,
        batch_first=True,
        init_token='<sos>',
        eos_token='<eos>',
        unk_token='<unk>'
    )
    
    if smiles_field_path is not None:
        suffix = '-'.join(properties)
        try:
            print(os.path.join(smiles_field_path, f'SRC_{suffix}.pkl'))
            SRC = pickle.load(open(os.path.join(smiles_field_path, 
                                                f'SRC_{suffix}.pkl'), 'rb'))
            TRG = pickle.load(open(os.path.join(smiles_field_path, 
                                                f'TRG_{suffix}.pkl'), 'rb'))
>>>>>>> fcf9ad655023b78d35671fb9106128a1df15153d
        except:
            print(">>> Files SRC.pkl/TRG.pkl not in:" + field_path)
            exit(1)

    return (SRC, TRG)   


# torch.float32 is equal to torch.float
def property_field(property_list):
    PROP = [data.Field(use_vocab=False, sequential=False, batch_first=True,
            dtype=torch.float32) for _ in property_list]
    return PROP


def get_iter_field(property_list, field_path=None):
    
    SRC, TRG = smiles_field(property_list, field_path)
    
    PROP = property_field(property_list)
    
    src_field = [('src', SRC)]
    src_field.extend([(f'src_{property_list[i]}', PROP[i])
                      for i in range(len(property_list))])
    trg_field = [('trg', TRG)]
    trg_field.extend([(f'trg_{property_list[i]}', PROP[i])
                      for i in range(len(property_list))])
    
    return src_field + trg_field, SRC, TRG


def untokenize(tokens, vocab):
    smi = ""
    for token in tokens:
        if token == vocab.stoi['<eos>']:
            break
        elif token != vocab.stoi['<sos>']:
            smi += vocab.itos[token]
    return smi


<<<<<<< HEAD
def id_to_smi(ids, TRG):
    """Convert ids into smiles based on TRG.
    
    Args:
        ids (List[int]): a list of integers representing a SMILES
        TRG (torchtext.data.Field): field for target

    Returns:
        smi (str): a string representing a molecule                 
    """
    smi = ''
    for i in ids:
        if i == TRG.vocab.stoi['<eos>']:
            break
        if i != TRG.vocab.stoi['<sos>']:
            smi += TRG.vocab.itos[i]
    return smi


def smi_to_id(smi, TRG, add_sos=False, add_eos=False):
    """Convert smiles into ids based on TRG.
    
    Args:
        smi (str): a string representing a molecule
        TRG (torchtext.data.Field): field for target

    Returns:
        ids (List[int]): a list of integers representing a SMILES
    """
    token = TRG.tokenize(smi)
    ids = []
    if add_sos:
        ids.append(TRG.vocab.stoi['<sos>'])
    ids.extend([TRG.vocab.stoi[t] for t in token])
    if add_eos:
        ids.append(TRG.vocab.stoi['<eos>'])    
    return ids

=======
>>>>>>> fcf9ad655023b78d35671fb9106128a1df15153d
